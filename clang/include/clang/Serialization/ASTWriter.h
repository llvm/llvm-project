//===--- ASTWriter.h - AST File Writer --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTWriter class, which writes an AST file
//  containing a serialized representation of a translation unit.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SERIALIZATION_ASTWRITER_H
#define LLVM_CLANG_SERIALIZATION_ASTWRITER_H

#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include <map>
#include <queue>
#include <vector>

namespace llvm {
  class APFloat;
  class APInt;
  class BitstreamWriter;
}

namespace clang {

class ASTContext;
class Attr;
class NestedNameSpecifier;
class CXXBaseSpecifier;
class CXXCtorInitializer;
class FileEntry;
class FPOptions;
class HeaderSearch;
class HeaderSearchOptions;
class IdentifierResolver;
class MacroDefinitionRecord;
class MacroDirective;
class MacroInfo;
class OpaqueValueExpr;
class OpenCLOptions;
class ASTReader;
class Module;
class ModuleFileExtension;
class ModuleFileExtensionWriter;
class PreprocessedEntity;
class PreprocessingRecord;
class Preprocessor;
class RecordDecl;
class Sema;
class SourceManager;
struct StoredDeclsList;
class SwitchCase;
class TargetInfo;
class Token;
class VersionTuple;
class ASTUnresolvedSet;

namespace SrcMgr { class SLocEntry; }

/// \brief Writes an AST file containing the contents of a translation unit.
///
/// The ASTWriter class produces a bitstream containing the serialized
/// representation of a given abstract syntax tree and its supporting
/// data structures. This bitstream can be de-serialized via an
/// instance of the ASTReader class.
class ASTWriter : public ASTDeserializationListener,
                  public ASTMutationListener {
public:
  typedef SmallVector<uint64_t, 64> RecordData;
  typedef SmallVectorImpl<uint64_t> RecordDataImpl;
  typedef ArrayRef<uint64_t> RecordDataRef;

  friend class ASTDeclWriter;
  friend class ASTStmtWriter;
private:
  /// \brief Map that provides the ID numbers of each type within the
  /// output stream, plus those deserialized from a chained PCH.
  ///
  /// The ID numbers of types are consecutive (in order of discovery)
  /// and start at 1. 0 is reserved for NULL. When types are actually
  /// stored in the stream, the ID number is shifted by 2 bits to
  /// allow for the const/volatile qualifiers.
  ///
  /// Keys in the map never have const/volatile qualifiers.
  typedef llvm::DenseMap<QualType, serialization::TypeIdx,
                         serialization::UnsafeQualTypeDenseMapInfo>
    TypeIdxMap;

  /// \brief The bitstream writer used to emit this precompiled header.
  llvm::BitstreamWriter &Stream;

  /// \brief The ASTContext we're writing.
  ASTContext *Context;

  /// \brief The preprocessor we're writing.
  Preprocessor *PP;

  /// \brief The reader of existing AST files, if we're chaining.
  ASTReader *Chain;

  /// \brief The module we're currently writing, if any.
  Module *WritingModule;

  /// \brief The base directory for any relative paths we emit.
  std::string BaseDirectory;

  /// \brief Indicates whether timestamps should be written to the produced
  /// module file. This is the case for files implicitly written to the
  /// module cache, where we need the timestamps to determine if the module
  /// file is up to date, but not otherwise.
  bool IncludeTimestamps;

  /// \brief Indicates when the AST writing is actively performing
  /// serialization, rather than just queueing updates.
  bool WritingAST;

  /// \brief Indicates that we are done serializing the collection of decls
  /// and types to emit.
  bool DoneWritingDeclsAndTypes;

  /// \brief Indicates that the AST contained compiler errors.
  bool ASTHasCompilerErrors;

  /// \brief Mapping from input file entries to the index into the
  /// offset table where information about that input file is stored.
  llvm::DenseMap<const FileEntry *, uint32_t> InputFileIDs;

  /// \brief Stores a declaration or a type to be written to the AST file.
  class DeclOrType {
  public:
    DeclOrType(Decl *D) : Stored(D), IsType(false) { }
    DeclOrType(QualType T) : Stored(T.getAsOpaquePtr()), IsType(true) { }

    bool isType() const { return IsType; }
    bool isDecl() const { return !IsType; }

    QualType getType() const {
      assert(isType() && "Not a type!");
      return QualType::getFromOpaquePtr(Stored);
    }

    Decl *getDecl() const {
      assert(isDecl() && "Not a decl!");
      return static_cast<Decl *>(Stored);
    }

  private:
    void *Stored;
    bool IsType;
  };

  /// \brief The declarations and types to emit.
  std::queue<DeclOrType> DeclTypesToEmit;

  /// \brief The first ID number we can use for our own declarations.
  serialization::DeclID FirstDeclID;

  /// \brief The decl ID that will be assigned to the next new decl.
  serialization::DeclID NextDeclID;

  /// \brief Map that provides the ID numbers of each declaration within
  /// the output stream, as well as those deserialized from a chained PCH.
  ///
  /// The ID numbers of declarations are consecutive (in order of
  /// discovery) and start at 2. 1 is reserved for the translation
  /// unit, while 0 is reserved for NULL.
  llvm::DenseMap<const Decl *, serialization::DeclID> DeclIDs;

  /// \brief Offset of each declaration in the bitstream, indexed by
  /// the declaration's ID.
  std::vector<serialization::DeclOffset> DeclOffsets;

  /// \brief Sorted (by file offset) vector of pairs of file offset/DeclID.
  typedef SmallVector<std::pair<unsigned, serialization::DeclID>, 64>
    LocDeclIDsTy;
  struct DeclIDInFileInfo {
    LocDeclIDsTy DeclIDs;
    /// \brief Set when the DeclIDs vectors from all files are joined, this
    /// indicates the index that this particular vector has in the global one.
    unsigned FirstDeclIndex;
  };
  typedef llvm::DenseMap<FileID, DeclIDInFileInfo *> FileDeclIDsTy;

  /// \brief Map from file SLocEntries to info about the file-level declarations
  /// that it contains.
  FileDeclIDsTy FileDeclIDs;

  void associateDeclWithFile(const Decl *D, serialization::DeclID);

  /// \brief The first ID number we can use for our own types.
  serialization::TypeID FirstTypeID;

  /// \brief The type ID that will be assigned to the next new type.
  serialization::TypeID NextTypeID;

  /// \brief Map that provides the ID numbers of each type within the
  /// output stream, plus those deserialized from a chained PCH.
  ///
  /// The ID numbers of types are consecutive (in order of discovery)
  /// and start at 1. 0 is reserved for NULL. When types are actually
  /// stored in the stream, the ID number is shifted by 2 bits to
  /// allow for the const/volatile qualifiers.
  ///
  /// Keys in the map never have const/volatile qualifiers.
  TypeIdxMap TypeIdxs;

  /// \brief Offset of each type in the bitstream, indexed by
  /// the type's ID.
  std::vector<uint32_t> TypeOffsets;

  /// \brief The first ID number we can use for our own identifiers.
  serialization::IdentID FirstIdentID;

  /// \brief The identifier ID that will be assigned to the next new identifier.
  serialization::IdentID NextIdentID;

  /// \brief Map that provides the ID numbers of each identifier in
  /// the output stream.
  ///
  /// The ID numbers for identifiers are consecutive (in order of
  /// discovery), starting at 1. An ID of zero refers to a NULL
  /// IdentifierInfo.
  llvm::MapVector<const IdentifierInfo *, serialization::IdentID> IdentifierIDs;

  /// \brief The first ID number we can use for our own macros.
  serialization::MacroID FirstMacroID;

  /// \brief The identifier ID that will be assigned to the next new identifier.
  serialization::MacroID NextMacroID;

  /// \brief Map that provides the ID numbers of each macro.
  llvm::DenseMap<MacroInfo *, serialization::MacroID> MacroIDs;

  struct MacroInfoToEmitData {
    const IdentifierInfo *Name;
    MacroInfo *MI;
    serialization::MacroID ID;
  };
  /// \brief The macro infos to emit.
  std::vector<MacroInfoToEmitData> MacroInfosToEmit;

  llvm::DenseMap<const IdentifierInfo *, uint64_t> IdentMacroDirectivesOffsetMap;

  /// @name FlushStmt Caches
  /// @{

  /// \brief Set of parent Stmts for the currently serializing sub-stmt.
  llvm::DenseSet<Stmt *> ParentStmts;

  /// \brief Offsets of sub-stmts already serialized. The offset points
  /// just after the stmt record.
  llvm::DenseMap<Stmt *, uint64_t> SubStmtEntries;

  /// @}

  /// \brief Offsets of each of the identifier IDs into the identifier
  /// table.
  std::vector<uint32_t> IdentifierOffsets;

  /// \brief The first ID number we can use for our own submodules.
  serialization::SubmoduleID FirstSubmoduleID;
  
  /// \brief The submodule ID that will be assigned to the next new submodule.
  serialization::SubmoduleID NextSubmoduleID;

  /// \brief The first ID number we can use for our own selectors.
  serialization::SelectorID FirstSelectorID;

  /// \brief The selector ID that will be assigned to the next new selector.
  serialization::SelectorID NextSelectorID;

  /// \brief Map that provides the ID numbers of each Selector.
  llvm::MapVector<Selector, serialization::SelectorID> SelectorIDs;

  /// \brief Offset of each selector within the method pool/selector
  /// table, indexed by the Selector ID (-1).
  std::vector<uint32_t> SelectorOffsets;

  /// \brief Mapping from macro definitions (as they occur in the preprocessing
  /// record) to the macro IDs.
  llvm::DenseMap<const MacroDefinitionRecord *,
                 serialization::PreprocessedEntityID> MacroDefinitions;

  /// \brief Cache of indices of anonymous declarations within their lexical
  /// contexts.
  llvm::DenseMap<const Decl *, unsigned> AnonymousDeclarationNumbers;

  /// An update to a Decl.
  class DeclUpdate {
    /// A DeclUpdateKind.
    unsigned Kind;
    union {
      const Decl *Dcl;
      void *Type;
      unsigned Loc;
      unsigned Val;
      Module *Mod;
      const Attr *Attribute;
    };

  public:
    DeclUpdate(unsigned Kind) : Kind(Kind), Dcl(nullptr) {}
    DeclUpdate(unsigned Kind, const Decl *Dcl) : Kind(Kind), Dcl(Dcl) {}
    DeclUpdate(unsigned Kind, QualType Type)
        : Kind(Kind), Type(Type.getAsOpaquePtr()) {}
    DeclUpdate(unsigned Kind, SourceLocation Loc)
        : Kind(Kind), Loc(Loc.getRawEncoding()) {}
    DeclUpdate(unsigned Kind, unsigned Val)
        : Kind(Kind), Val(Val) {}
    DeclUpdate(unsigned Kind, Module *M)
          : Kind(Kind), Mod(M) {}
    DeclUpdate(unsigned Kind, const Attr *Attribute)
          : Kind(Kind), Attribute(Attribute) {}

    unsigned getKind() const { return Kind; }
    const Decl *getDecl() const { return Dcl; }
    QualType getType() const { return QualType::getFromOpaquePtr(Type); }
    SourceLocation getLoc() const {
      return SourceLocation::getFromRawEncoding(Loc);
    }
    unsigned getNumber() const { return Val; }
    Module *getModule() const { return Mod; }
    const Attr *getAttr() const { return Attribute; }
  };

  typedef SmallVector<DeclUpdate, 1> UpdateRecord;
  typedef llvm::MapVector<const Decl *, UpdateRecord> DeclUpdateMap;
  /// \brief Mapping from declarations that came from a chained PCH to the
  /// record containing modifications to them.
  DeclUpdateMap DeclUpdates;

  typedef llvm::DenseMap<Decl *, Decl *> FirstLatestDeclMap;
  /// \brief Map of first declarations from a chained PCH that point to the
  /// most recent declarations in another PCH.
  FirstLatestDeclMap FirstLatestDecls;

  /// \brief Declarations encountered that might be external
  /// definitions.
  ///
  /// We keep track of external definitions and other 'interesting' declarations
  /// as we are emitting declarations to the AST file. The AST file contains a
  /// separate record for these declarations, which are provided to the AST
  /// consumer by the AST reader. This is behavior is required to properly cope with,
  /// e.g., tentative variable definitions that occur within
  /// headers. The declarations themselves are stored as declaration
  /// IDs, since they will be written out to an EAGERLY_DESERIALIZED_DECLS
  /// record.
  SmallVector<uint64_t, 16> EagerlyDeserializedDecls;

  /// \brief DeclContexts that have received extensions since their serialized
  /// form.
  ///
  /// For namespaces, when we're chaining and encountering a namespace, we check
  /// if its primary namespace comes from the chain. If it does, we add the
  /// primary to this set, so that we can write out lexical content updates for
  /// it.
  llvm::SmallSetVector<const DeclContext *, 16> UpdatedDeclContexts;

  /// \brief Keeps track of visible decls that were added in DeclContexts
  /// coming from another AST file.
  SmallVector<const Decl *, 16> UpdatingVisibleDecls;

  /// \brief The set of Objective-C class that have categories we
  /// should serialize.
  llvm::SetVector<ObjCInterfaceDecl *> ObjCClassesWithCategories;
                    
  struct ReplacedDeclInfo {
    serialization::DeclID ID;
    uint64_t Offset;
    unsigned Loc;

    ReplacedDeclInfo() : ID(0), Offset(0), Loc(0) {}
    ReplacedDeclInfo(serialization::DeclID ID, uint64_t Offset,
                     SourceLocation Loc)
      : ID(ID), Offset(Offset), Loc(Loc.getRawEncoding()) {}
  };

  /// \brief Decls that have been replaced in the current dependent AST file.
  ///
  /// When a decl changes fundamentally after being deserialized (this shouldn't
  /// happen, but the ObjC AST nodes are designed this way), it will be
  /// serialized again. In this case, it is registered here, so that the reader
  /// knows to read the updated version.
  SmallVector<ReplacedDeclInfo, 16> ReplacedDecls;
                 
  /// \brief The set of declarations that may have redeclaration chains that
  /// need to be serialized.
  llvm::SmallVector<const Decl *, 16> Redeclarations;

  /// \brief A cache of the first local declaration for "interesting"
  /// redeclaration chains.
  llvm::DenseMap<const Decl *, const Decl *> FirstLocalDeclCache;
                                      
  /// \brief Statements that we've encountered while serializing a
  /// declaration or type.
  SmallVector<Stmt *, 16> StmtsToEmit;

  /// \brief Statements collection to use for ASTWriter::AddStmt().
  /// It will point to StmtsToEmit unless it is overriden.
  SmallVector<Stmt *, 16> *CollectedStmts;

  /// \brief Mapping from SwitchCase statements to IDs.
  llvm::DenseMap<SwitchCase *, unsigned> SwitchCaseIDs;

  /// \brief The number of statements written to the AST file.
  unsigned NumStatements;

  /// \brief The number of macros written to the AST file.
  unsigned NumMacros;

  /// \brief The number of lexical declcontexts written to the AST
  /// file.
  unsigned NumLexicalDeclContexts;

  /// \brief The number of visible declcontexts written to the AST
  /// file.
  unsigned NumVisibleDeclContexts;

  /// \brief The offset of each CXXBaseSpecifier set within the AST.
  SmallVector<uint32_t, 16> CXXBaseSpecifiersOffsets;

  /// \brief The first ID number we can use for our own base specifiers.
  serialization::CXXBaseSpecifiersID FirstCXXBaseSpecifiersID;

  /// \brief The base specifiers ID that will be assigned to the next new
  /// set of C++ base specifiers.
  serialization::CXXBaseSpecifiersID NextCXXBaseSpecifiersID;

  /// \brief A set of C++ base specifiers that is queued to be written into the
  /// AST file.
  struct QueuedCXXBaseSpecifiers {
    QueuedCXXBaseSpecifiers() : ID(), Bases(), BasesEnd() { }

    QueuedCXXBaseSpecifiers(serialization::CXXBaseSpecifiersID ID,
                            CXXBaseSpecifier const *Bases,
                            CXXBaseSpecifier const *BasesEnd)
      : ID(ID), Bases(Bases), BasesEnd(BasesEnd) { }

    serialization::CXXBaseSpecifiersID ID;
    CXXBaseSpecifier const * Bases;
    CXXBaseSpecifier const * BasesEnd;
  };

  /// \brief Queue of C++ base specifiers to be written to the AST file,
  /// in the order they should be written.
  SmallVector<QueuedCXXBaseSpecifiers, 2> CXXBaseSpecifiersToWrite;

  /// \brief The offset of each CXXCtorInitializer list within the AST.
  SmallVector<uint32_t, 16> CXXCtorInitializersOffsets;

  /// \brief The first ID number we can use for our own ctor initializers.
  serialization::CXXCtorInitializersID FirstCXXCtorInitializersID;

  /// \brief The ctor initializers ID that will be assigned to the next new
  /// list of C++ ctor initializers.
  serialization::CXXCtorInitializersID NextCXXCtorInitializersID;

  /// \brief A set of C++ ctor initializers that is queued to be written
  /// into the AST file.
  struct QueuedCXXCtorInitializers {
    QueuedCXXCtorInitializers() : ID() {}

    QueuedCXXCtorInitializers(serialization::CXXCtorInitializersID ID,
                              ArrayRef<CXXCtorInitializer*> Inits)
        : ID(ID), Inits(Inits) {}

    serialization::CXXCtorInitializersID ID;
    ArrayRef<CXXCtorInitializer*> Inits;
  };

  /// \brief Queue of C++ ctor initializers to be written to the AST file,
  /// in the order they should be written.
  SmallVector<QueuedCXXCtorInitializers, 2> CXXCtorInitializersToWrite;

  /// \brief A mapping from each known submodule to its ID number, which will
  /// be a positive integer.
  llvm::DenseMap<Module *, unsigned> SubmoduleIDs;

  /// \brief A list of the module file extension writers.
  std::vector<std::unique_ptr<ModuleFileExtensionWriter>>
    ModuleFileExtensionWriters;

  /// \brief Retrieve or create a submodule ID for this module.
  unsigned getSubmoduleID(Module *Mod);

  /// \brief Write the given subexpression to the bitstream.
  void WriteSubStmt(Stmt *S,
                    llvm::DenseMap<Stmt *, uint64_t> &SubStmtEntries,
                    llvm::DenseSet<Stmt *> &ParentStmts);

  void WriteBlockInfoBlock();
  uint64_t WriteControlBlock(Preprocessor &PP, ASTContext &Context,
                             StringRef isysroot, const std::string &OutputFile);
  void WriteInputFiles(SourceManager &SourceMgr, HeaderSearchOptions &HSOpts,
                       bool Modules);
  void WriteSourceManagerBlock(SourceManager &SourceMgr,
                               const Preprocessor &PP);
  void WritePreprocessor(const Preprocessor &PP, bool IsModule);
  void WriteHeaderSearch(const HeaderSearch &HS);
  void WritePreprocessorDetail(PreprocessingRecord &PPRec);
  void WriteSubmodules(Module *WritingModule);
                                        
  void WritePragmaDiagnosticMappings(const DiagnosticsEngine &Diag,
                                     bool isModule);
  void WriteCXXBaseSpecifiersOffsets();
  void WriteCXXCtorInitializersOffsets();

  unsigned TypeExtQualAbbrev;
  unsigned TypeFunctionProtoAbbrev;
  void WriteTypeAbbrevs();
  void WriteType(QualType T);

  bool isLookupResultExternal(StoredDeclsList &Result, DeclContext *DC);
  bool isLookupResultEntirelyExternal(StoredDeclsList &Result, DeclContext *DC);

  void GenerateNameLookupTable(const DeclContext *DC,
                               llvm::SmallVectorImpl<char> &LookupTable);
  uint64_t WriteDeclContextLexicalBlock(ASTContext &Context, DeclContext *DC);
  uint64_t WriteDeclContextVisibleBlock(ASTContext &Context, DeclContext *DC);
  void WriteTypeDeclOffsets();
  void WriteFileDeclIDsMap();
  void WriteComments();
  void WriteSelectors(Sema &SemaRef);
  void WriteReferencedSelectorsPool(Sema &SemaRef);
  void WriteIdentifierTable(Preprocessor &PP, IdentifierResolver &IdResolver,
                            bool IsModule);
  void WriteAttributes(ArrayRef<const Attr*> Attrs, RecordDataImpl &Record);
  void WriteDeclUpdatesBlocks(RecordDataImpl &OffsetsRecord);
  void WriteDeclReplacementsBlock();
  void WriteDeclContextVisibleUpdate(const DeclContext *DC);
  void WriteFPPragmaOptions(const FPOptions &Opts);
  void WriteOpenCLExtensions(Sema &SemaRef);
  void WriteObjCCategories();
  void WriteLateParsedTemplates(Sema &SemaRef);
  void WriteOptimizePragmaOptions(Sema &SemaRef);
  void WriteModuleFileExtension(Sema &SemaRef,
                                ModuleFileExtensionWriter &Writer);

  unsigned DeclParmVarAbbrev;
  unsigned DeclContextLexicalAbbrev;
  unsigned DeclContextVisibleLookupAbbrev;
  unsigned UpdateVisibleAbbrev;
  unsigned DeclRecordAbbrev;
  unsigned DeclTypedefAbbrev;
  unsigned DeclVarAbbrev;
  unsigned DeclFieldAbbrev;
  unsigned DeclEnumAbbrev;
  unsigned DeclObjCIvarAbbrev;
  unsigned DeclCXXMethodAbbrev;

  unsigned DeclRefExprAbbrev;
  unsigned CharacterLiteralAbbrev;
  unsigned IntegerLiteralAbbrev;
  unsigned ExprImplicitCastAbbrev;

  void WriteDeclAbbrevs();
  void WriteDecl(ASTContext &Context, Decl *D);
  void AddFunctionDefinition(const FunctionDecl *FD, RecordData &Record);

  uint64_t WriteASTCore(Sema &SemaRef,
                        StringRef isysroot, const std::string &OutputFile,
                        Module *WritingModule);

public:
  /// \brief Create a new precompiled header writer that outputs to
  /// the given bitstream.
  ASTWriter(llvm::BitstreamWriter &Stream,
            ArrayRef<llvm::IntrusiveRefCntPtr<ModuleFileExtension>> Extensions,
            bool IncludeTimestamps = true);
  ~ASTWriter() override;

  const LangOptions &getLangOpts() const;

  /// \brief Get a timestamp for output into the AST file. The actual timestamp
  /// of the specified file may be ignored if we have been instructed to not
  /// include timestamps in the output file.
  time_t getTimestampForOutput(const FileEntry *E) const;

  /// \brief Write a precompiled header for the given semantic analysis.
  ///
  /// \param SemaRef a reference to the semantic analysis object that processed
  /// the AST to be written into the precompiled header.
  ///
  /// \param WritingModule The module that we are writing. If null, we are
  /// writing a precompiled header.
  ///
  /// \param isysroot if non-empty, write a relocatable file whose headers
  /// are relative to the given system root. If we're writing a module, its
  /// build directory will be used in preference to this if both are available.
  ///
  /// \return the module signature, which eventually will be a hash of
  /// the module but currently is merely a random 32-bit number.
  uint64_t WriteAST(Sema &SemaRef, const std::string &OutputFile,
                    Module *WritingModule, StringRef isysroot,
                    bool hasErrors = false);

  /// \brief Emit a token.
  void AddToken(const Token &Tok, RecordDataImpl &Record);

  /// \brief Emit a source location.
  void AddSourceLocation(SourceLocation Loc, RecordDataImpl &Record);

  /// \brief Emit a source range.
  void AddSourceRange(SourceRange Range, RecordDataImpl &Record);

  /// \brief Emit an integral value.
  void AddAPInt(const llvm::APInt &Value, RecordDataImpl &Record);

  /// \brief Emit a signed integral value.
  void AddAPSInt(const llvm::APSInt &Value, RecordDataImpl &Record);

  /// \brief Emit a floating-point value.
  void AddAPFloat(const llvm::APFloat &Value, RecordDataImpl &Record);

  /// \brief Emit a reference to an identifier.
  void AddIdentifierRef(const IdentifierInfo *II, RecordDataImpl &Record);

  /// \brief Emit a Selector (which is a smart pointer reference).
  void AddSelectorRef(Selector, RecordDataImpl &Record);

  /// \brief Emit a CXXTemporary.
  void AddCXXTemporary(const CXXTemporary *Temp, RecordDataImpl &Record);

  /// \brief Emit a set of C++ base specifiers to the record.
  void AddCXXBaseSpecifiersRef(CXXBaseSpecifier const *Bases,
                               CXXBaseSpecifier const *BasesEnd,
                               RecordDataImpl &Record);

  /// \brief Get the unique number used to refer to the given selector.
  serialization::SelectorID getSelectorRef(Selector Sel);

  /// \brief Get the unique number used to refer to the given identifier.
  serialization::IdentID getIdentifierRef(const IdentifierInfo *II);

  /// \brief Get the unique number used to refer to the given macro.
  serialization::MacroID getMacroRef(MacroInfo *MI, const IdentifierInfo *Name);

  /// \brief Determine the ID of an already-emitted macro.
  serialization::MacroID getMacroID(MacroInfo *MI);

  uint64_t getMacroDirectivesOffset(const IdentifierInfo *Name);

  /// \brief Emit a reference to a type.
  void AddTypeRef(QualType T, RecordDataImpl &Record);

  /// \brief Force a type to be emitted and get its ID.
  serialization::TypeID GetOrCreateTypeID(QualType T);

  /// \brief Determine the type ID of an already-emitted type.
  serialization::TypeID getTypeID(QualType T) const;

  /// \brief Emits a reference to a declarator info.
  void AddTypeSourceInfo(TypeSourceInfo *TInfo, RecordDataImpl &Record);

  /// \brief Emits a type with source-location information.
  void AddTypeLoc(TypeLoc TL, RecordDataImpl &Record);

  /// \brief Emits a template argument location info.
  void AddTemplateArgumentLocInfo(TemplateArgument::ArgKind Kind,
                                  const TemplateArgumentLocInfo &Arg,
                                  RecordDataImpl &Record);

  /// \brief Emits a template argument location.
  void AddTemplateArgumentLoc(const TemplateArgumentLoc &Arg,
                              RecordDataImpl &Record);

  /// \brief Emits an AST template argument list info.
  void AddASTTemplateArgumentListInfo(
                          const ASTTemplateArgumentListInfo *ASTTemplArgList,
                          RecordDataImpl &Record);

  /// \brief Find the first local declaration of a given local redeclarable
  /// decl.
  const Decl *getFirstLocalDecl(const Decl *D);

  /// \brief Emit a reference to a declaration.
  void AddDeclRef(const Decl *D, RecordDataImpl &Record);


  /// \brief Force a declaration to be emitted and get its ID.
  serialization::DeclID GetDeclRef(const Decl *D);

  /// \brief Determine the declaration ID of an already-emitted
  /// declaration.
  serialization::DeclID getDeclID(const Decl *D);

  /// \brief Emit a declaration name.
  void AddDeclarationName(DeclarationName Name, RecordDataImpl &Record);
  void AddDeclarationNameLoc(const DeclarationNameLoc &DNLoc,
                             DeclarationName Name, RecordDataImpl &Record);
  void AddDeclarationNameInfo(const DeclarationNameInfo &NameInfo,
                              RecordDataImpl &Record);
  unsigned getAnonymousDeclarationNumber(const NamedDecl *D);

  void AddQualifierInfo(const QualifierInfo &Info, RecordDataImpl &Record);

  /// \brief Emit a nested name specifier.
  void AddNestedNameSpecifier(NestedNameSpecifier *NNS, RecordDataImpl &Record);

  /// \brief Emit a nested name specifier with source-location information.
  void AddNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS,
                                 RecordDataImpl &Record);

  /// \brief Emit a template name.
  void AddTemplateName(TemplateName Name, RecordDataImpl &Record);

  /// \brief Emit a template argument.
  void AddTemplateArgument(const TemplateArgument &Arg, RecordDataImpl &Record);

  /// \brief Emit a template parameter list.
  void AddTemplateParameterList(const TemplateParameterList *TemplateParams,
                                RecordDataImpl &Record);

  /// \brief Emit a template argument list.
  void AddTemplateArgumentList(const TemplateArgumentList *TemplateArgs,
                                RecordDataImpl &Record);

  /// \brief Emit a UnresolvedSet structure.
  void AddUnresolvedSet(const ASTUnresolvedSet &Set, RecordDataImpl &Record);

  /// \brief Emit a C++ base specifier.
  void AddCXXBaseSpecifier(const CXXBaseSpecifier &Base,
                           RecordDataImpl &Record);

  /// \brief Emit the ID for a CXXCtorInitializer array and register the array
  /// for later serialization.
  void AddCXXCtorInitializersRef(ArrayRef<CXXCtorInitializer *> Inits,
                                 RecordDataImpl &Record);

  /// \brief Emit a CXXCtorInitializer array.
  void AddCXXCtorInitializers(
                             const CXXCtorInitializer * const *CtorInitializers,
                             unsigned NumCtorInitializers,
                             RecordDataImpl &Record);

  void AddCXXDefinitionData(const CXXRecordDecl *D, RecordDataImpl &Record);

  /// \brief Add a string to the given record.
  void AddString(StringRef Str, RecordDataImpl &Record);

  /// \brief Convert a path from this build process into one that is appropriate
  /// for emission in the module file.
  bool PreparePathForOutput(SmallVectorImpl<char> &Path);

  /// \brief Add a path to the given record.
  void AddPath(StringRef Path, RecordDataImpl &Record);

  /// \brief Emit the current record with the given path as a blob.
  void EmitRecordWithPath(unsigned Abbrev, RecordDataRef Record,
                          StringRef Path);

  /// \brief Add a version tuple to the given record
  void AddVersionTuple(const VersionTuple &Version, RecordDataImpl &Record);

  /// \brief Infer the submodule ID that contains an entity at the given
  /// source location.
  serialization::SubmoduleID inferSubmoduleIDFromLocation(SourceLocation Loc);

  /// \brief Retrieve or create a submodule ID for this module, or return 0 if
  /// the submodule is neither local (a submodle of the currently-written module)
  /// nor from an imported module.
  unsigned getLocalOrImportedSubmoduleID(Module *Mod);

  /// \brief Note that the identifier II occurs at the given offset
  /// within the identifier table.
  void SetIdentifierOffset(const IdentifierInfo *II, uint32_t Offset);

  /// \brief Note that the selector Sel occurs at the given offset
  /// within the method pool/selector table.
  void SetSelectorOffset(Selector Sel, uint32_t Offset);

  /// \brief Add the given statement or expression to the queue of
  /// statements to emit.
  ///
  /// This routine should be used when emitting types and declarations
  /// that have expressions as part of their formulation. Once the
  /// type or declaration has been written, call FlushStmts() to write
  /// the corresponding statements just after the type or
  /// declaration.
  void AddStmt(Stmt *S) {
      CollectedStmts->push_back(S);
  }

  /// \brief Flush all of the statements and expressions that have
  /// been added to the queue via AddStmt().
  void FlushStmts();

  /// \brief Flush all of the C++ base specifier sets that have been added
  /// via \c AddCXXBaseSpecifiersRef().
  void FlushCXXBaseSpecifiers();

  /// \brief Flush all of the C++ constructor initializer lists that have been
  /// added via \c AddCXXCtorInitializersRef().
  void FlushCXXCtorInitializers();

  /// \brief Flush all pending records that are tacked onto the end of
  /// decl and decl update records.
  void FlushPendingAfterDecl() {
    FlushStmts();
    FlushCXXBaseSpecifiers();
    FlushCXXCtorInitializers();
  }

  /// \brief Record an ID for the given switch-case statement.
  unsigned RecordSwitchCaseID(SwitchCase *S);

  /// \brief Retrieve the ID for the given switch-case statement.
  unsigned getSwitchCaseID(SwitchCase *S);

  void ClearSwitchCaseIDs();

  unsigned getTypeExtQualAbbrev() const {
    return TypeExtQualAbbrev;
  }
  unsigned getTypeFunctionProtoAbbrev() const {
    return TypeFunctionProtoAbbrev;
  }

  unsigned getDeclParmVarAbbrev() const { return DeclParmVarAbbrev; }
  unsigned getDeclRecordAbbrev() const { return DeclRecordAbbrev; }
  unsigned getDeclTypedefAbbrev() const { return DeclTypedefAbbrev; }
  unsigned getDeclVarAbbrev() const { return DeclVarAbbrev; }
  unsigned getDeclFieldAbbrev() const { return DeclFieldAbbrev; }
  unsigned getDeclEnumAbbrev() const { return DeclEnumAbbrev; }
  unsigned getDeclObjCIvarAbbrev() const { return DeclObjCIvarAbbrev; }
  unsigned getDeclCXXMethodAbbrev() const { return DeclCXXMethodAbbrev; }

  unsigned getDeclRefExprAbbrev() const { return DeclRefExprAbbrev; }
  unsigned getCharacterLiteralAbbrev() const { return CharacterLiteralAbbrev; }
  unsigned getIntegerLiteralAbbrev() const { return IntegerLiteralAbbrev; }
  unsigned getExprImplicitCastAbbrev() const { return ExprImplicitCastAbbrev; }

  bool hasChain() const { return Chain; }
  ASTReader *getChain() const { return Chain; }

  // ASTDeserializationListener implementation
  void ReaderInitialized(ASTReader *Reader) override;
  void IdentifierRead(serialization::IdentID ID, IdentifierInfo *II) override;
  void MacroRead(serialization::MacroID ID, MacroInfo *MI) override;
  void TypeRead(serialization::TypeIdx Idx, QualType T) override;
  void SelectorRead(serialization::SelectorID ID, Selector Sel) override;
  void MacroDefinitionRead(serialization::PreprocessedEntityID ID,
                           MacroDefinitionRecord *MD) override;
  void ModuleRead(serialization::SubmoduleID ID, Module *Mod) override;

  // ASTMutationListener implementation.
  void CompletedTagDefinition(const TagDecl *D) override;
  void AddedVisibleDecl(const DeclContext *DC, const Decl *D) override;
  void AddedCXXImplicitMember(const CXXRecordDecl *RD, const Decl *D) override;
  void ResolvedExceptionSpec(const FunctionDecl *FD) override;
  void DeducedReturnType(const FunctionDecl *FD, QualType ReturnType) override;
  void ResolvedOperatorDelete(const CXXDestructorDecl *DD,
                              const FunctionDecl *Delete) override;
  void CompletedImplicitDefinition(const FunctionDecl *D) override;
  void StaticDataMemberInstantiated(const VarDecl *D) override;
  void DefaultArgumentInstantiated(const ParmVarDecl *D) override;
  void FunctionDefinitionInstantiated(const FunctionDecl *D) override;
  void AddedObjCCategoryToInterface(const ObjCCategoryDecl *CatD,
                                    const ObjCInterfaceDecl *IFD) override;
  void DeclarationMarkedUsed(const Decl *D) override;
  void DeclarationMarkedOpenMPThreadPrivate(const Decl *D) override;
  void RedefinedHiddenDefinition(const NamedDecl *D, Module *M) override;
  void AddedAttributeToRecord(const Attr *Attr,
                              const RecordDecl *Record) override;
};

/// \brief AST and semantic-analysis consumer that generates a
/// precompiled header from the parsed source code.
class PCHGenerator : public SemaConsumer {
  const Preprocessor &PP;
  std::string OutputFile;
  clang::Module *Module;
  std::string isysroot;
  Sema *SemaPtr;
  std::shared_ptr<PCHBuffer> Buffer;
  llvm::BitstreamWriter Stream;
  ASTWriter Writer;
  bool AllowASTWithErrors;

protected:
  ASTWriter &getWriter() { return Writer; }
  const ASTWriter &getWriter() const { return Writer; }
  SmallVectorImpl<char> &getPCH() const { return Buffer->Data; }

public:
  PCHGenerator(
    const Preprocessor &PP, StringRef OutputFile,
    clang::Module *Module, StringRef isysroot,
    std::shared_ptr<PCHBuffer> Buffer,
    ArrayRef<llvm::IntrusiveRefCntPtr<ModuleFileExtension>> Extensions,
    bool AllowASTWithErrors = false,
    bool IncludeTimestamps = true);
  ~PCHGenerator() override;
  void InitializeSema(Sema &S) override { SemaPtr = &S; }
  void HandleTranslationUnit(ASTContext &Ctx) override;
  ASTMutationListener *GetASTMutationListener() override;
  ASTDeserializationListener *GetASTDeserializationListener() override;
  bool hasEmittedPCH() const { return Buffer->IsComplete; }
};

} // end namespace clang

#endif
