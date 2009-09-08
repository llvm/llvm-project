//===--- PCHWriter.h - Precompiled Headers Writer ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHWriter class, which writes a precompiled header.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/PCHWriter.h"
#include "../Sema/Sema.h" // FIXME: move header into include/clang/Sema
#include "../Sema/IdentifierResolver.h" // FIXME: move header 
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/OnDiskHashTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManagerInternals.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Path.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// Type serialization
//===----------------------------------------------------------------------===//

namespace {
  class VISIBILITY_HIDDEN PCHTypeWriter {
    PCHWriter &Writer;
    PCHWriter::RecordData &Record;

  public:
    /// \brief Type code that corresponds to the record generated.
    pch::TypeCode Code;

    PCHTypeWriter(PCHWriter &Writer, PCHWriter::RecordData &Record) 
      : Writer(Writer), Record(Record), Code(pch::TYPE_EXT_QUAL) { }

    void VisitArrayType(const ArrayType *T);
    void VisitFunctionType(const FunctionType *T);
    void VisitTagType(const TagType *T);

#define TYPE(Class, Base) void Visit##Class##Type(const Class##Type *T);
#define ABSTRACT_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
  };
}

void PCHTypeWriter::VisitExtQualType(const ExtQualType *T) {
  Writer.AddTypeRef(QualType(T->getBaseType(), 0), Record);
  Record.push_back(T->getObjCGCAttr()); // FIXME: use stable values
  Record.push_back(T->getAddressSpace());
  Code = pch::TYPE_EXT_QUAL;
}

void PCHTypeWriter::VisitBuiltinType(const BuiltinType *T) {
  assert(false && "Built-in types are never serialized");
}

void PCHTypeWriter::VisitFixedWidthIntType(const FixedWidthIntType *T) {
  Record.push_back(T->getWidth());
  Record.push_back(T->isSigned());
  Code = pch::TYPE_FIXED_WIDTH_INT;
}

void PCHTypeWriter::VisitComplexType(const ComplexType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Code = pch::TYPE_COMPLEX;
}

void PCHTypeWriter::VisitPointerType(const PointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_POINTER;
}

void PCHTypeWriter::VisitBlockPointerType(const BlockPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);  
  Code = pch::TYPE_BLOCK_POINTER;
}

void PCHTypeWriter::VisitLValueReferenceType(const LValueReferenceType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_LVALUE_REFERENCE;
}

void PCHTypeWriter::VisitRValueReferenceType(const RValueReferenceType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_RVALUE_REFERENCE;
}

void PCHTypeWriter::VisitMemberPointerType(const MemberPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);  
  Writer.AddTypeRef(QualType(T->getClass(), 0), Record);  
  Code = pch::TYPE_MEMBER_POINTER;
}

void PCHTypeWriter::VisitArrayType(const ArrayType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Record.push_back(T->getSizeModifier()); // FIXME: stable values
  Record.push_back(T->getIndexTypeQualifier()); // FIXME: stable values
}

void PCHTypeWriter::VisitConstantArrayType(const ConstantArrayType *T) {
  VisitArrayType(T);
  Writer.AddAPInt(T->getSize(), Record);
  Code = pch::TYPE_CONSTANT_ARRAY;
}

void PCHTypeWriter
::VisitConstantArrayWithExprType(const ConstantArrayWithExprType *T) {
  VisitArrayType(T);
  Writer.AddSourceLocation(T->getLBracketLoc(), Record);
  Writer.AddSourceLocation(T->getRBracketLoc(), Record);
  Writer.AddAPInt(T->getSize(), Record);
  Writer.AddStmt(T->getSizeExpr());
  Code = pch::TYPE_CONSTANT_ARRAY_WITH_EXPR;
}

void PCHTypeWriter
::VisitConstantArrayWithoutExprType(const ConstantArrayWithoutExprType *T) {
  VisitArrayType(T);
  Writer.AddAPInt(T->getSize(), Record);
  Code = pch::TYPE_CONSTANT_ARRAY_WITHOUT_EXPR;
}

void PCHTypeWriter::VisitIncompleteArrayType(const IncompleteArrayType *T) {
  VisitArrayType(T);
  Code = pch::TYPE_INCOMPLETE_ARRAY;
}

void PCHTypeWriter::VisitVariableArrayType(const VariableArrayType *T) {
  VisitArrayType(T);
  Writer.AddSourceLocation(T->getLBracketLoc(), Record);
  Writer.AddSourceLocation(T->getRBracketLoc(), Record);
  Writer.AddStmt(T->getSizeExpr());
  Code = pch::TYPE_VARIABLE_ARRAY;
}

void PCHTypeWriter::VisitVectorType(const VectorType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Record.push_back(T->getNumElements());
  Code = pch::TYPE_VECTOR;
}

void PCHTypeWriter::VisitExtVectorType(const ExtVectorType *T) {
  VisitVectorType(T);
  Code = pch::TYPE_EXT_VECTOR;
}

void PCHTypeWriter::VisitFunctionType(const FunctionType *T) {
  Writer.AddTypeRef(T->getResultType(), Record);
}

void PCHTypeWriter::VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
  VisitFunctionType(T);
  Code = pch::TYPE_FUNCTION_NO_PROTO;
}

void PCHTypeWriter::VisitFunctionProtoType(const FunctionProtoType *T) {
  VisitFunctionType(T);
  Record.push_back(T->getNumArgs());
  for (unsigned I = 0, N = T->getNumArgs(); I != N; ++I)
    Writer.AddTypeRef(T->getArgType(I), Record);
  Record.push_back(T->isVariadic());
  Record.push_back(T->getTypeQuals());
  Record.push_back(T->hasExceptionSpec());
  Record.push_back(T->hasAnyExceptionSpec());
  Record.push_back(T->getNumExceptions());
  for (unsigned I = 0, N = T->getNumExceptions(); I != N; ++I)
    Writer.AddTypeRef(T->getExceptionType(I), Record);
  Code = pch::TYPE_FUNCTION_PROTO;
}

void PCHTypeWriter::VisitTypedefType(const TypedefType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  Code = pch::TYPE_TYPEDEF;
}

void PCHTypeWriter::VisitTypeOfExprType(const TypeOfExprType *T) {
  Writer.AddStmt(T->getUnderlyingExpr());
  Code = pch::TYPE_TYPEOF_EXPR;
}

void PCHTypeWriter::VisitTypeOfType(const TypeOfType *T) {
  Writer.AddTypeRef(T->getUnderlyingType(), Record);
  Code = pch::TYPE_TYPEOF;
}

void PCHTypeWriter::VisitDecltypeType(const DecltypeType *T) {
  Writer.AddStmt(T->getUnderlyingExpr());
  Code = pch::TYPE_DECLTYPE;
}

void PCHTypeWriter::VisitTagType(const TagType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  assert(!T->isBeingDefined() && 
         "Cannot serialize in the middle of a type definition");
}

void PCHTypeWriter::VisitRecordType(const RecordType *T) {
  VisitTagType(T);
  Code = pch::TYPE_RECORD;
}

void PCHTypeWriter::VisitEnumType(const EnumType *T) {
  VisitTagType(T);
  Code = pch::TYPE_ENUM;
}

void 
PCHTypeWriter::VisitTemplateSpecializationType(
                                       const TemplateSpecializationType *T) {
  // FIXME: Serialize this type (C++ only)
  assert(false && "Cannot serialize template specialization types");
}

void PCHTypeWriter::VisitQualifiedNameType(const QualifiedNameType *T) {
  // FIXME: Serialize this type (C++ only)
  assert(false && "Cannot serialize qualified name types");
}

void PCHTypeWriter::VisitObjCInterfaceType(const ObjCInterfaceType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  Record.push_back(T->getNumProtocols());
  for (ObjCInterfaceType::qual_iterator I = T->qual_begin(),
       E = T->qual_end(); I != E; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = pch::TYPE_OBJC_INTERFACE;
}

void
PCHTypeWriter::VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);  
  Record.push_back(T->getNumProtocols());
  for (ObjCInterfaceType::qual_iterator I = T->qual_begin(),
       E = T->qual_end(); I != E; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = pch::TYPE_OBJC_OBJECT_POINTER;
}

//===----------------------------------------------------------------------===//
// PCHWriter Implementation
//===----------------------------------------------------------------------===//

static void EmitBlockID(unsigned ID, const char *Name,
                        llvm::BitstreamWriter &Stream,
                        PCHWriter::RecordData &Record) {
  Record.clear();
  Record.push_back(ID);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETBID, Record);

  // Emit the block name if present.
  if (Name == 0 || Name[0] == 0) return;
  Record.clear();
  while (*Name)
    Record.push_back(*Name++);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_BLOCKNAME, Record);
}

static void EmitRecordID(unsigned ID, const char *Name,
                         llvm::BitstreamWriter &Stream,
                         PCHWriter::RecordData &Record) {
  Record.clear();
  Record.push_back(ID);
  while (*Name)
    Record.push_back(*Name++);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETRECORDNAME, Record);
}

static void AddStmtsExprs(llvm::BitstreamWriter &Stream,
                          PCHWriter::RecordData &Record) {
#define RECORD(X) EmitRecordID(pch::X, #X, Stream, Record)
  RECORD(STMT_STOP);
  RECORD(STMT_NULL_PTR);
  RECORD(STMT_NULL);
  RECORD(STMT_COMPOUND);
  RECORD(STMT_CASE);
  RECORD(STMT_DEFAULT);
  RECORD(STMT_LABEL);
  RECORD(STMT_IF);
  RECORD(STMT_SWITCH);
  RECORD(STMT_WHILE);
  RECORD(STMT_DO);
  RECORD(STMT_FOR);
  RECORD(STMT_GOTO);
  RECORD(STMT_INDIRECT_GOTO);
  RECORD(STMT_CONTINUE);
  RECORD(STMT_BREAK);
  RECORD(STMT_RETURN);
  RECORD(STMT_DECL);
  RECORD(STMT_ASM);
  RECORD(EXPR_PREDEFINED);
  RECORD(EXPR_DECL_REF);
  RECORD(EXPR_INTEGER_LITERAL);
  RECORD(EXPR_FLOATING_LITERAL);
  RECORD(EXPR_IMAGINARY_LITERAL);
  RECORD(EXPR_STRING_LITERAL);
  RECORD(EXPR_CHARACTER_LITERAL);
  RECORD(EXPR_PAREN);
  RECORD(EXPR_UNARY_OPERATOR);
  RECORD(EXPR_SIZEOF_ALIGN_OF);
  RECORD(EXPR_ARRAY_SUBSCRIPT);
  RECORD(EXPR_CALL);
  RECORD(EXPR_MEMBER);
  RECORD(EXPR_BINARY_OPERATOR);
  RECORD(EXPR_COMPOUND_ASSIGN_OPERATOR);
  RECORD(EXPR_CONDITIONAL_OPERATOR);
  RECORD(EXPR_IMPLICIT_CAST);
  RECORD(EXPR_CSTYLE_CAST);
  RECORD(EXPR_COMPOUND_LITERAL);
  RECORD(EXPR_EXT_VECTOR_ELEMENT);
  RECORD(EXPR_INIT_LIST);
  RECORD(EXPR_DESIGNATED_INIT);
  RECORD(EXPR_IMPLICIT_VALUE_INIT);
  RECORD(EXPR_VA_ARG);
  RECORD(EXPR_ADDR_LABEL);
  RECORD(EXPR_STMT);
  RECORD(EXPR_TYPES_COMPATIBLE);
  RECORD(EXPR_CHOOSE);
  RECORD(EXPR_GNU_NULL);
  RECORD(EXPR_SHUFFLE_VECTOR);
  RECORD(EXPR_BLOCK);
  RECORD(EXPR_BLOCK_DECL_REF);
  RECORD(EXPR_OBJC_STRING_LITERAL);
  RECORD(EXPR_OBJC_ENCODE);
  RECORD(EXPR_OBJC_SELECTOR_EXPR);
  RECORD(EXPR_OBJC_PROTOCOL_EXPR);
  RECORD(EXPR_OBJC_IVAR_REF_EXPR);
  RECORD(EXPR_OBJC_PROPERTY_REF_EXPR);
  RECORD(EXPR_OBJC_KVC_REF_EXPR);
  RECORD(EXPR_OBJC_MESSAGE_EXPR);
  RECORD(EXPR_OBJC_SUPER_EXPR);
  RECORD(STMT_OBJC_FOR_COLLECTION);
  RECORD(STMT_OBJC_CATCH);
  RECORD(STMT_OBJC_FINALLY);
  RECORD(STMT_OBJC_AT_TRY);
  RECORD(STMT_OBJC_AT_SYNCHRONIZED);
  RECORD(STMT_OBJC_AT_THROW);
#undef RECORD
}
  
void PCHWriter::WriteBlockInfoBlock() {
  RecordData Record;
  Stream.EnterSubblock(llvm::bitc::BLOCKINFO_BLOCK_ID, 3);
  
#define BLOCK(X) EmitBlockID(pch::X ## _ID, #X, Stream, Record)
#define RECORD(X) EmitRecordID(pch::X, #X, Stream, Record)
 
  // PCH Top-Level Block.
  BLOCK(PCH_BLOCK);
  RECORD(ORIGINAL_FILE_NAME);
  RECORD(TYPE_OFFSET);
  RECORD(DECL_OFFSET);
  RECORD(LANGUAGE_OPTIONS);
  RECORD(METADATA);
  RECORD(IDENTIFIER_OFFSET);
  RECORD(IDENTIFIER_TABLE);
  RECORD(EXTERNAL_DEFINITIONS);
  RECORD(SPECIAL_TYPES);
  RECORD(STATISTICS);
  RECORD(TENTATIVE_DEFINITIONS);
  RECORD(LOCALLY_SCOPED_EXTERNAL_DECLS);
  RECORD(SELECTOR_OFFSETS);
  RECORD(METHOD_POOL);
  RECORD(PP_COUNTER_VALUE);
  RECORD(SOURCE_LOCATION_OFFSETS);
  RECORD(SOURCE_LOCATION_PRELOADS);
  RECORD(STAT_CACHE);
  RECORD(EXT_VECTOR_DECLS);
  RECORD(COMMENT_RANGES);
  
  // SourceManager Block.
  BLOCK(SOURCE_MANAGER_BLOCK);
  RECORD(SM_SLOC_FILE_ENTRY);
  RECORD(SM_SLOC_BUFFER_ENTRY);
  RECORD(SM_SLOC_BUFFER_BLOB);
  RECORD(SM_SLOC_INSTANTIATION_ENTRY);
  RECORD(SM_LINE_TABLE);
  RECORD(SM_HEADER_FILE_INFO);
  
  // Preprocessor Block.
  BLOCK(PREPROCESSOR_BLOCK);
  RECORD(PP_MACRO_OBJECT_LIKE);
  RECORD(PP_MACRO_FUNCTION_LIKE);
  RECORD(PP_TOKEN);

  // Types block.
  BLOCK(TYPES_BLOCK);
  RECORD(TYPE_EXT_QUAL);
  RECORD(TYPE_FIXED_WIDTH_INT);
  RECORD(TYPE_COMPLEX);
  RECORD(TYPE_POINTER);
  RECORD(TYPE_BLOCK_POINTER);
  RECORD(TYPE_LVALUE_REFERENCE);
  RECORD(TYPE_RVALUE_REFERENCE);
  RECORD(TYPE_MEMBER_POINTER);
  RECORD(TYPE_CONSTANT_ARRAY);
  RECORD(TYPE_INCOMPLETE_ARRAY);
  RECORD(TYPE_VARIABLE_ARRAY);
  RECORD(TYPE_VECTOR);
  RECORD(TYPE_EXT_VECTOR);
  RECORD(TYPE_FUNCTION_PROTO);
  RECORD(TYPE_FUNCTION_NO_PROTO);
  RECORD(TYPE_TYPEDEF);
  RECORD(TYPE_TYPEOF_EXPR);
  RECORD(TYPE_TYPEOF);
  RECORD(TYPE_RECORD);
  RECORD(TYPE_ENUM);
  RECORD(TYPE_OBJC_INTERFACE);
  RECORD(TYPE_OBJC_OBJECT_POINTER);
  // Statements and Exprs can occur in the Types block.
  AddStmtsExprs(Stream, Record);

  // Decls block.
  BLOCK(DECLS_BLOCK);
  RECORD(DECL_ATTR);
  RECORD(DECL_TRANSLATION_UNIT);
  RECORD(DECL_TYPEDEF);
  RECORD(DECL_ENUM);
  RECORD(DECL_RECORD);
  RECORD(DECL_ENUM_CONSTANT);
  RECORD(DECL_FUNCTION);
  RECORD(DECL_OBJC_METHOD);
  RECORD(DECL_OBJC_INTERFACE);
  RECORD(DECL_OBJC_PROTOCOL);
  RECORD(DECL_OBJC_IVAR);
  RECORD(DECL_OBJC_AT_DEFS_FIELD);
  RECORD(DECL_OBJC_CLASS);
  RECORD(DECL_OBJC_FORWARD_PROTOCOL);
  RECORD(DECL_OBJC_CATEGORY);
  RECORD(DECL_OBJC_CATEGORY_IMPL);
  RECORD(DECL_OBJC_IMPLEMENTATION);
  RECORD(DECL_OBJC_COMPATIBLE_ALIAS);
  RECORD(DECL_OBJC_PROPERTY);
  RECORD(DECL_OBJC_PROPERTY_IMPL);
  RECORD(DECL_FIELD);
  RECORD(DECL_VAR);
  RECORD(DECL_IMPLICIT_PARAM);
  RECORD(DECL_PARM_VAR);
  RECORD(DECL_ORIGINAL_PARM_VAR);
  RECORD(DECL_FILE_SCOPE_ASM);
  RECORD(DECL_BLOCK);
  RECORD(DECL_CONTEXT_LEXICAL);
  RECORD(DECL_CONTEXT_VISIBLE);
  // Statements and Exprs can occur in the Decls block.
  AddStmtsExprs(Stream, Record);
#undef RECORD
#undef BLOCK
  Stream.ExitBlock();
}

/// \brief Adjusts the given filename to only write out the portion of the
/// filename that is not part of the system root directory.
/// 
/// \param Filename the file name to adjust.
///
/// \param isysroot When non-NULL, the PCH file is a relocatable PCH file and
/// the returned filename will be adjusted by this system root.
///
/// \returns either the original filename (if it needs no adjustment) or the
/// adjusted filename (which points into the @p Filename parameter).
static const char * 
adjustFilenameForRelocatablePCH(const char *Filename, const char *isysroot) {
  assert(Filename && "No file name to adjust?");
  
  if (!isysroot)
    return Filename;
  
  // Verify that the filename and the system root have the same prefix.
  unsigned Pos = 0;
  for (; Filename[Pos] && isysroot[Pos]; ++Pos)
    if (Filename[Pos] != isysroot[Pos])
      return Filename; // Prefixes don't match.
  
  // We hit the end of the filename before we hit the end of the system root.
  if (!Filename[Pos])
    return Filename;
  
  // If the file name has a '/' at the current position, skip over the '/'.
  // We distinguish sysroot-based includes from absolute includes by the
  // absence of '/' at the beginning of sysroot-based includes.
  if (Filename[Pos] == '/')
    ++Pos;
  
  return Filename + Pos;
}

/// \brief Write the PCH metadata (e.g., i686-apple-darwin9).
void PCHWriter::WriteMetadata(ASTContext &Context, const char *isysroot) {
  using namespace llvm;

  // Metadata
  const TargetInfo &Target = Context.Target;
  BitCodeAbbrev *MetaAbbrev = new BitCodeAbbrev();
  MetaAbbrev->Add(BitCodeAbbrevOp(pch::METADATA));
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // PCH major
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // PCH minor
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Clang major
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Clang minor
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Relocatable
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Target triple
  unsigned MetaAbbrevCode = Stream.EmitAbbrev(MetaAbbrev);
  
  RecordData Record;
  Record.push_back(pch::METADATA);
  Record.push_back(pch::VERSION_MAJOR);
  Record.push_back(pch::VERSION_MINOR);
  Record.push_back(CLANG_VERSION_MAJOR);
  Record.push_back(CLANG_VERSION_MINOR);
  Record.push_back(isysroot != 0);
  const char *Triple = Target.getTargetTriple();
  Stream.EmitRecordWithBlob(MetaAbbrevCode, Record, Triple, strlen(Triple));
  
  // Original file name
  SourceManager &SM = Context.getSourceManager();
  if (const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID())) {
    BitCodeAbbrev *FileAbbrev = new BitCodeAbbrev();
    FileAbbrev->Add(BitCodeAbbrevOp(pch::ORIGINAL_FILE_NAME));
    FileAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // File name
    unsigned FileAbbrevCode = Stream.EmitAbbrev(FileAbbrev);

    llvm::sys::Path MainFilePath(MainFile->getName());
    std::string MainFileName;
  
    if (!MainFilePath.isAbsolute()) {
      llvm::sys::Path P = llvm::sys::Path::GetCurrentDirectory();
      P.appendComponent(MainFilePath.toString());
      MainFileName = P.toString();
    } else {
      MainFileName = MainFilePath.toString();
    }

    const char *MainFileNameStr = MainFileName.c_str();
    MainFileNameStr = adjustFilenameForRelocatablePCH(MainFileNameStr, 
                                                      isysroot);
    RecordData Record;
    Record.push_back(pch::ORIGINAL_FILE_NAME);
    Stream.EmitRecordWithBlob(FileAbbrevCode, Record, MainFileNameStr,
                              strlen(MainFileNameStr));
  }
}

/// \brief Write the LangOptions structure.
void PCHWriter::WriteLanguageOptions(const LangOptions &LangOpts) {
  RecordData Record;
  Record.push_back(LangOpts.Trigraphs);
  Record.push_back(LangOpts.BCPLComment);  // BCPL-style '//' comments.
  Record.push_back(LangOpts.DollarIdents);  // '$' allowed in identifiers.
  Record.push_back(LangOpts.AsmPreprocessor);  // Preprocessor in asm mode.
  Record.push_back(LangOpts.GNUMode);  // True in gnu99 mode false in c99 mode (etc)
  Record.push_back(LangOpts.ImplicitInt);  // C89 implicit 'int'.
  Record.push_back(LangOpts.Digraphs);  // C94, C99 and C++
  Record.push_back(LangOpts.HexFloats);  // C99 Hexadecimal float constants.
  Record.push_back(LangOpts.C99);  // C99 Support
  Record.push_back(LangOpts.Microsoft);  // Microsoft extensions.
  Record.push_back(LangOpts.CPlusPlus);  // C++ Support
  Record.push_back(LangOpts.CPlusPlus0x);  // C++0x Support
  Record.push_back(LangOpts.CXXOperatorNames);  // Treat C++ operator names as keywords.
    
  Record.push_back(LangOpts.ObjC1);  // Objective-C 1 support enabled.
  Record.push_back(LangOpts.ObjC2);  // Objective-C 2 support enabled.
  Record.push_back(LangOpts.ObjCNonFragileABI);  // Objective-C modern abi enabled
    
  Record.push_back(LangOpts.PascalStrings);  // Allow Pascal strings
  Record.push_back(LangOpts.WritableStrings);  // Allow writable strings
  Record.push_back(LangOpts.LaxVectorConversions);
  Record.push_back(LangOpts.AltiVec);
  Record.push_back(LangOpts.Exceptions);  // Support exception handling.

  Record.push_back(LangOpts.NeXTRuntime); // Use NeXT runtime.
  Record.push_back(LangOpts.Freestanding); // Freestanding implementation
  Record.push_back(LangOpts.NoBuiltin); // Do not use builtin functions (-fno-builtin)

  // Whether static initializers are protected by locks.
  Record.push_back(LangOpts.ThreadsafeStatics);
  Record.push_back(LangOpts.Blocks); // block extension to C
  Record.push_back(LangOpts.EmitAllDecls); // Emit all declarations, even if
                                  // they are unused.
  Record.push_back(LangOpts.MathErrno); // Math functions must respect errno
                                  // (modulo the platform support).

  Record.push_back(LangOpts.OverflowChecking); // Extension to call a handler function when
                                  // signed integer arithmetic overflows.

  Record.push_back(LangOpts.HeinousExtensions); // Extensions that we really don't like and
                                  // may be ripped out at any time.

  Record.push_back(LangOpts.Optimize); // Whether __OPTIMIZE__ should be defined.
  Record.push_back(LangOpts.OptimizeSize); // Whether __OPTIMIZE_SIZE__ should be 
                                  // defined.
  Record.push_back(LangOpts.Static); // Should __STATIC__ be defined (as
                                  // opposed to __DYNAMIC__).
  Record.push_back(LangOpts.PICLevel); // The value for __PIC__, if non-zero.

  Record.push_back(LangOpts.GNUInline); // Should GNU inline semantics be
                                  // used (instead of C99 semantics).
  Record.push_back(LangOpts.NoInline); // Should __NO_INLINE__ be defined.
  Record.push_back(LangOpts.AccessControl); // Whether C++ access control should
                                            // be enabled.
  Record.push_back(LangOpts.CharIsSigned); // Whether char is a signed or
                                           // unsigned type
  Record.push_back(LangOpts.getGCMode());
  Record.push_back(LangOpts.getVisibilityMode());
  Record.push_back(LangOpts.InstantiationDepth);
  Record.push_back(LangOpts.OpenCL);
  Stream.EmitRecord(pch::LANGUAGE_OPTIONS, Record);
}

//===----------------------------------------------------------------------===//
// stat cache Serialization
//===----------------------------------------------------------------------===//

namespace {
// Trait used for the on-disk hash table of stat cache results.
class VISIBILITY_HIDDEN PCHStatCacheTrait {
public:
  typedef const char * key_type;
  typedef key_type key_type_ref;
  
  typedef std::pair<int, struct stat> data_type;
  typedef const data_type& data_type_ref;

  static unsigned ComputeHash(const char *path) {
    return BernsteinHash(path);
  }
  
  std::pair<unsigned,unsigned> 
    EmitKeyDataLength(llvm::raw_ostream& Out, const char *path,
                      data_type_ref Data) {
    unsigned StrLen = strlen(path);
    clang::io::Emit16(Out, StrLen);
    unsigned DataLen = 1; // result value
    if (Data.first == 0)
      DataLen += 4 + 4 + 2 + 8 + 8;
    clang::io::Emit8(Out, DataLen);
    return std::make_pair(StrLen + 1, DataLen);
  }
  
  void EmitKey(llvm::raw_ostream& Out, const char *path, unsigned KeyLen) {
    Out.write(path, KeyLen);
  }
  
  void EmitData(llvm::raw_ostream& Out, key_type_ref,
                data_type_ref Data, unsigned DataLen) {
    using namespace clang::io;
    uint64_t Start = Out.tell(); (void)Start;
    
    // Result of stat()
    Emit8(Out, Data.first? 1 : 0);
    
    if (Data.first == 0) {
      Emit32(Out, (uint32_t) Data.second.st_ino);
      Emit32(Out, (uint32_t) Data.second.st_dev);
      Emit16(Out, (uint16_t) Data.second.st_mode);
      Emit64(Out, (uint64_t) Data.second.st_mtime);
      Emit64(Out, (uint64_t) Data.second.st_size);
    }

    assert(Out.tell() - Start == DataLen && "Wrong data length");
  }
};
} // end anonymous namespace

/// \brief Write the stat() system call cache to the PCH file.
void PCHWriter::WriteStatCache(MemorizeStatCalls &StatCalls,
                               const char *isysroot) {
  // Build the on-disk hash table containing information about every
  // stat() call.
  OnDiskChainedHashTableGenerator<PCHStatCacheTrait> Generator;
  unsigned NumStatEntries = 0;
  for (MemorizeStatCalls::iterator Stat = StatCalls.begin(), 
                                StatEnd = StatCalls.end();
       Stat != StatEnd; ++Stat, ++NumStatEntries) {
    const char *Filename = Stat->first();
    Filename = adjustFilenameForRelocatablePCH(Filename, isysroot);
    Generator.insert(Filename, Stat->second);
  }
  
  // Create the on-disk hash table in a buffer.
  llvm::SmallVector<char, 4096> StatCacheData; 
  uint32_t BucketOffset;
  {
    llvm::raw_svector_ostream Out(StatCacheData);
    // Make sure that no bucket is at offset 0
    clang::io::Emit32(Out, 0);
    BucketOffset = Generator.Emit(Out);
  }

  // Create a blob abbreviation
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::STAT_CACHE));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned StatCacheAbbrev = Stream.EmitAbbrev(Abbrev);

  // Write the stat cache
  RecordData Record;
  Record.push_back(pch::STAT_CACHE);
  Record.push_back(BucketOffset);
  Record.push_back(NumStatEntries);
  Stream.EmitRecordWithBlob(StatCacheAbbrev, Record, 
                            &StatCacheData.front(), 
                            StatCacheData.size());
}

//===----------------------------------------------------------------------===//
// Source Manager Serialization
//===----------------------------------------------------------------------===//

/// \brief Create an abbreviation for the SLocEntry that refers to a
/// file.
static unsigned CreateSLocFileAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SM_SLOC_FILE_ENTRY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Include location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // Characteristic
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Line directives
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // File name
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Create an abbreviation for the SLocEntry that refers to a
/// buffer.
static unsigned CreateSLocBufferAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SM_SLOC_BUFFER_ENTRY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Include location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // Characteristic
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Line directives
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Buffer name blob
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Create an abbreviation for the SLocEntry that refers to a
/// buffer's blob.
static unsigned CreateSLocBufferBlobAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SM_SLOC_BUFFER_BLOB));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Blob
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Create an abbreviation for the SLocEntry that refers to an
/// buffer.
static unsigned CreateSLocInstantiationAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SM_SLOC_INSTANTIATION_ENTRY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Spelling location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Start location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // End location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Token length
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Writes the block containing the serialized form of the
/// source manager.
///
/// TODO: We should probably use an on-disk hash table (stored in a
/// blob), indexed based on the file name, so that we only create
/// entries for files that we actually need. In the common case (no
/// errors), we probably won't have to create file entries for any of
/// the files in the AST.
void PCHWriter::WriteSourceManagerBlock(SourceManager &SourceMgr,
                                        const Preprocessor &PP,
                                        const char *isysroot) {
  RecordData Record;

  // Enter the source manager block.
  Stream.EnterSubblock(pch::SOURCE_MANAGER_BLOCK_ID, 3);

  // Abbreviations for the various kinds of source-location entries.
  unsigned SLocFileAbbrv = CreateSLocFileAbbrev(Stream);
  unsigned SLocBufferAbbrv = CreateSLocBufferAbbrev(Stream);
  unsigned SLocBufferBlobAbbrv = CreateSLocBufferBlobAbbrev(Stream);
  unsigned SLocInstantiationAbbrv = CreateSLocInstantiationAbbrev(Stream);

  // Write the line table.
  if (SourceMgr.hasLineTable()) {
    LineTableInfo &LineTable = SourceMgr.getLineTable();

    // Emit the file names
    Record.push_back(LineTable.getNumFilenames());
    for (unsigned I = 0, N = LineTable.getNumFilenames(); I != N; ++I) {
      // Emit the file name
      const char *Filename = LineTable.getFilename(I);
      Filename = adjustFilenameForRelocatablePCH(Filename, isysroot);
      unsigned FilenameLen = Filename? strlen(Filename) : 0;
      Record.push_back(FilenameLen);
      if (FilenameLen)
        Record.insert(Record.end(), Filename, Filename + FilenameLen);
    }
    
    // Emit the line entries
    for (LineTableInfo::iterator L = LineTable.begin(), LEnd = LineTable.end();
         L != LEnd; ++L) {
      // Emit the file ID
      Record.push_back(L->first);
      
      // Emit the line entries
      Record.push_back(L->second.size());
      for (std::vector<LineEntry>::iterator LE = L->second.begin(), 
                                         LEEnd = L->second.end();
           LE != LEEnd; ++LE) {
        Record.push_back(LE->FileOffset);
        Record.push_back(LE->LineNo);
        Record.push_back(LE->FilenameID);
        Record.push_back((unsigned)LE->FileKind);
        Record.push_back(LE->IncludeOffset);
      }
    }
    Stream.EmitRecord(pch::SM_LINE_TABLE, Record);
  }

  // Write out entries for all of the header files we know about.
  HeaderSearch &HS = PP.getHeaderSearchInfo();  
  Record.clear();
  for (HeaderSearch::header_file_iterator I = HS.header_file_begin(), 
                                          E = HS.header_file_end();
       I != E; ++I) {
    Record.push_back(I->isImport);
    Record.push_back(I->DirInfo);
    Record.push_back(I->NumIncludes);
    AddIdentifierRef(I->ControllingMacro, Record);
    Stream.EmitRecord(pch::SM_HEADER_FILE_INFO, Record);
    Record.clear();
  }

  // Write out the source location entry table. We skip the first
  // entry, which is always the same dummy entry.
  std::vector<uint32_t> SLocEntryOffsets;
  RecordData PreloadSLocs;
  SLocEntryOffsets.reserve(SourceMgr.sloc_entry_size() - 1);
  for (SourceManager::sloc_entry_iterator 
         SLoc = SourceMgr.sloc_entry_begin() + 1,
         SLocEnd = SourceMgr.sloc_entry_end();
       SLoc != SLocEnd; ++SLoc) {
    // Record the offset of this source-location entry.
    SLocEntryOffsets.push_back(Stream.GetCurrentBitNo());

    // Figure out which record code to use.
    unsigned Code;
    if (SLoc->isFile()) {
      if (SLoc->getFile().getContentCache()->Entry)
        Code = pch::SM_SLOC_FILE_ENTRY;
      else
        Code = pch::SM_SLOC_BUFFER_ENTRY;
    } else
      Code = pch::SM_SLOC_INSTANTIATION_ENTRY;
    Record.clear();
    Record.push_back(Code);

    Record.push_back(SLoc->getOffset());
    if (SLoc->isFile()) {
      const SrcMgr::FileInfo &File = SLoc->getFile();
      Record.push_back(File.getIncludeLoc().getRawEncoding());
      Record.push_back(File.getFileCharacteristic()); // FIXME: stable encoding
      Record.push_back(File.hasLineDirectives());

      const SrcMgr::ContentCache *Content = File.getContentCache();
      if (Content->Entry) {
        // The source location entry is a file. The blob associated
        // with this entry is the file name.
        
        // Turn the file name into an absolute path, if it isn't already.
        const char *Filename = Content->Entry->getName();
        llvm::sys::Path FilePath(Filename, strlen(Filename));
        std::string FilenameStr;
        if (!FilePath.isAbsolute()) {
          llvm::sys::Path P = llvm::sys::Path::GetCurrentDirectory();
          P.appendComponent(FilePath.toString());
          FilenameStr = P.toString();
          Filename = FilenameStr.c_str();
        }
        
        Filename = adjustFilenameForRelocatablePCH(Filename, isysroot);
        Stream.EmitRecordWithBlob(SLocFileAbbrv, Record, Filename, 
                                  strlen(Filename));

        // FIXME: For now, preload all file source locations, so that
        // we get the appropriate File entries in the reader. This is
        // a temporary measure.
        PreloadSLocs.push_back(SLocEntryOffsets.size());
      } else {
        // The source location entry is a buffer. The blob associated
        // with this entry contains the contents of the buffer.

        // We add one to the size so that we capture the trailing NULL
        // that is required by llvm::MemoryBuffer::getMemBuffer (on
        // the reader side).
        const llvm::MemoryBuffer *Buffer = Content->getBuffer();
        const char *Name = Buffer->getBufferIdentifier();
        Stream.EmitRecordWithBlob(SLocBufferAbbrv, Record, Name, strlen(Name) + 1);
        Record.clear();
        Record.push_back(pch::SM_SLOC_BUFFER_BLOB);
        Stream.EmitRecordWithBlob(SLocBufferBlobAbbrv, Record,
                             Buffer->getBufferStart(),
                             Buffer->getBufferSize() + 1);

        if (strcmp(Name, "<built-in>") == 0)
          PreloadSLocs.push_back(SLocEntryOffsets.size());
      }
    } else {
      // The source location entry is an instantiation.
      const SrcMgr::InstantiationInfo &Inst = SLoc->getInstantiation();
      Record.push_back(Inst.getSpellingLoc().getRawEncoding());
      Record.push_back(Inst.getInstantiationLocStart().getRawEncoding());
      Record.push_back(Inst.getInstantiationLocEnd().getRawEncoding());

      // Compute the token length for this macro expansion.
      unsigned NextOffset = SourceMgr.getNextOffset();
      SourceManager::sloc_entry_iterator NextSLoc = SLoc;
      if (++NextSLoc != SLocEnd)
        NextOffset = NextSLoc->getOffset();
      Record.push_back(NextOffset - SLoc->getOffset() - 1);
      Stream.EmitRecordWithAbbrev(SLocInstantiationAbbrv, Record);
    }
  }

  Stream.ExitBlock();

  if (SLocEntryOffsets.empty())
    return;

  // Write the source-location offsets table into the PCH block. This
  // table is used for lazily loading source-location information.
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SOURCE_LOCATION_OFFSETS));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 16)); // # of slocs
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 16)); // next offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // offsets
  unsigned SLocOffsetsAbbrev = Stream.EmitAbbrev(Abbrev);
  
  Record.clear();
  Record.push_back(pch::SOURCE_LOCATION_OFFSETS);
  Record.push_back(SLocEntryOffsets.size());
  Record.push_back(SourceMgr.getNextOffset());
  Stream.EmitRecordWithBlob(SLocOffsetsAbbrev, Record,
                            (const char *)&SLocEntryOffsets.front(), 
                           SLocEntryOffsets.size()*sizeof(SLocEntryOffsets[0]));

  // Write the source location entry preloads array, telling the PCH
  // reader which source locations entries it should load eagerly.
  Stream.EmitRecord(pch::SOURCE_LOCATION_PRELOADS, PreloadSLocs);
}

//===----------------------------------------------------------------------===//
// Preprocessor Serialization
//===----------------------------------------------------------------------===//

/// \brief Writes the block containing the serialized form of the
/// preprocessor.
///
void PCHWriter::WritePreprocessor(const Preprocessor &PP) {
  RecordData Record;

  // If the preprocessor __COUNTER__ value has been bumped, remember it.
  if (PP.getCounterValue() != 0) {
    Record.push_back(PP.getCounterValue());
    Stream.EmitRecord(pch::PP_COUNTER_VALUE, Record);
    Record.clear();
  }

  // Enter the preprocessor block.
  Stream.EnterSubblock(pch::PREPROCESSOR_BLOCK_ID, 2);
  
  // If the PCH file contains __DATE__ or __TIME__ emit a warning about this.
  // FIXME: use diagnostics subsystem for localization etc.
  if (PP.SawDateOrTime())
    fprintf(stderr, "warning: precompiled header used __DATE__ or __TIME__.\n");
    
  // Loop over all the macro definitions that are live at the end of the file,
  // emitting each to the PP section.
  for (Preprocessor::macro_iterator I = PP.macro_begin(), E = PP.macro_end();
       I != E; ++I) {
    // FIXME: This emits macros in hash table order, we should do it in a stable
    // order so that output is reproducible.
    MacroInfo *MI = I->second;

    // Don't emit builtin macros like __LINE__ to the PCH file unless they have
    // been redefined by the header (in which case they are not isBuiltinMacro).
    if (MI->isBuiltinMacro())
      continue;

    // FIXME: Remove this identifier reference?
    AddIdentifierRef(I->first, Record);
    MacroOffsets[I->first] = Stream.GetCurrentBitNo();
    Record.push_back(MI->getDefinitionLoc().getRawEncoding());
    Record.push_back(MI->isUsed());
    
    unsigned Code;
    if (MI->isObjectLike()) {
      Code = pch::PP_MACRO_OBJECT_LIKE;
    } else {
      Code = pch::PP_MACRO_FUNCTION_LIKE;
      
      Record.push_back(MI->isC99Varargs());
      Record.push_back(MI->isGNUVarargs());
      Record.push_back(MI->getNumArgs());
      for (MacroInfo::arg_iterator I = MI->arg_begin(), E = MI->arg_end();
           I != E; ++I)
        AddIdentifierRef(*I, Record);
    }
    Stream.EmitRecord(Code, Record);
    Record.clear();

    // Emit the tokens array.
    for (unsigned TokNo = 0, e = MI->getNumTokens(); TokNo != e; ++TokNo) {
      // Note that we know that the preprocessor does not have any annotation
      // tokens in it because they are created by the parser, and thus can't be
      // in a macro definition.
      const Token &Tok = MI->getReplacementToken(TokNo);
      
      Record.push_back(Tok.getLocation().getRawEncoding());
      Record.push_back(Tok.getLength());

      // FIXME: When reading literal tokens, reconstruct the literal pointer if
      // it is needed.
      AddIdentifierRef(Tok.getIdentifierInfo(), Record);
      
      // FIXME: Should translate token kind to a stable encoding.
      Record.push_back(Tok.getKind());
      // FIXME: Should translate token flags to a stable encoding.
      Record.push_back(Tok.getFlags());
      
      Stream.EmitRecord(pch::PP_TOKEN, Record);
      Record.clear();
    }
    ++NumMacros;
  }
  Stream.ExitBlock();
}

void PCHWriter::WriteComments(ASTContext &Context) {
  using namespace llvm;
  
  if (Context.Comments.empty())
    return;
  
  BitCodeAbbrev *CommentAbbrev = new BitCodeAbbrev();
  CommentAbbrev->Add(BitCodeAbbrevOp(pch::COMMENT_RANGES));
  CommentAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned CommentCode = Stream.EmitAbbrev(CommentAbbrev);
  
  RecordData Record;
  Record.push_back(pch::COMMENT_RANGES);
  Stream.EmitRecordWithBlob(CommentCode, Record, 
                            (const char*)&Context.Comments[0],
                            Context.Comments.size() * sizeof(SourceRange));
}

//===----------------------------------------------------------------------===//
// Type Serialization
//===----------------------------------------------------------------------===//

/// \brief Write the representation of a type to the PCH stream.
void PCHWriter::WriteType(const Type *T) {
  pch::TypeID &ID = TypeIDs[T];
  if (ID == 0) // we haven't seen this type before.
    ID = NextTypeID++;
  
  // Record the offset for this type.
  if (TypeOffsets.size() == ID - pch::NUM_PREDEF_TYPE_IDS)
    TypeOffsets.push_back(Stream.GetCurrentBitNo());
  else if (TypeOffsets.size() < ID - pch::NUM_PREDEF_TYPE_IDS) {
    TypeOffsets.resize(ID + 1 - pch::NUM_PREDEF_TYPE_IDS);
    TypeOffsets[ID - pch::NUM_PREDEF_TYPE_IDS] = Stream.GetCurrentBitNo();
  }

  RecordData Record;
  
  // Emit the type's representation.
  PCHTypeWriter W(*this, Record);
  switch (T->getTypeClass()) {
    // For all of the concrete, non-dependent types, call the
    // appropriate visitor function.
#define TYPE(Class, Base) \
    case Type::Class: W.Visit##Class##Type(cast<Class##Type>(T)); break;
#define ABSTRACT_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"

    // For all of the dependent type nodes (which only occur in C++
    // templates), produce an error.
#define TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    assert(false && "Cannot serialize dependent type nodes");
    break;
  }

  // Emit the serialized record.
  Stream.EmitRecord(W.Code, Record);

  // Flush any expressions that were written as part of this type.
  FlushStmts();
}

/// \brief Write a block containing all of the types.
void PCHWriter::WriteTypesBlock(ASTContext &Context) {
  // Enter the types block.
  Stream.EnterSubblock(pch::TYPES_BLOCK_ID, 2);

  // Emit all of the types that need to be emitted (so far).
  while (!TypesToEmit.empty()) {
    const Type *T = TypesToEmit.front();
    TypesToEmit.pop();
    assert(!isa<BuiltinType>(T) && "Built-in types are not serialized");
    WriteType(T);
  }

  // Exit the types block
  Stream.ExitBlock();
}

//===----------------------------------------------------------------------===//
// Declaration Serialization
//===----------------------------------------------------------------------===//

/// \brief Write the block containing all of the declaration IDs
/// lexically declared within the given DeclContext.
///
/// \returns the offset of the DECL_CONTEXT_LEXICAL block within the
/// bistream, or 0 if no block was written.
uint64_t PCHWriter::WriteDeclContextLexicalBlock(ASTContext &Context, 
                                                 DeclContext *DC) {
  if (DC->decls_empty())
    return 0;

  uint64_t Offset = Stream.GetCurrentBitNo();
  RecordData Record;
  for (DeclContext::decl_iterator D = DC->decls_begin(), DEnd = DC->decls_end();
         D != DEnd; ++D)
    AddDeclRef(*D, Record);

  ++NumLexicalDeclContexts;
  Stream.EmitRecord(pch::DECL_CONTEXT_LEXICAL, Record);
  return Offset;
}

/// \brief Write the block containing all of the declaration IDs
/// visible from the given DeclContext.
///
/// \returns the offset of the DECL_CONTEXT_VISIBLE block within the
/// bistream, or 0 if no block was written.
uint64_t PCHWriter::WriteDeclContextVisibleBlock(ASTContext &Context,
                                                 DeclContext *DC) {
  if (DC->getPrimaryContext() != DC)
    return 0;

  // Since there is no name lookup into functions or methods, and we
  // perform name lookup for the translation unit via the
  // IdentifierInfo chains, don't bother to build a
  // visible-declarations table for these entities.
  if (DC->isFunctionOrMethod() || DC->isTranslationUnit())
    return 0;

  // Force the DeclContext to build a its name-lookup table.
  DC->lookup(DeclarationName());

  // Serialize the contents of the mapping used for lookup. Note that,
  // although we have two very different code paths, the serialized
  // representation is the same for both cases: a declaration name,
  // followed by a size, followed by references to the visible
  // declarations that have that name.
  uint64_t Offset = Stream.GetCurrentBitNo();
  RecordData Record;
  StoredDeclsMap *Map = static_cast<StoredDeclsMap*>(DC->getLookupPtr());
  if (!Map)
    return 0;

  for (StoredDeclsMap::iterator D = Map->begin(), DEnd = Map->end();
       D != DEnd; ++D) {
    AddDeclarationName(D->first, Record);
    DeclContext::lookup_result Result = D->second.getLookupResult(Context);
    Record.push_back(Result.second - Result.first);
    for(; Result.first != Result.second; ++Result.first)
      AddDeclRef(*Result.first, Record);
  }

  if (Record.size() == 0)
    return 0;

  Stream.EmitRecord(pch::DECL_CONTEXT_VISIBLE, Record);
  ++NumVisibleDeclContexts;
  return Offset;
}

//===----------------------------------------------------------------------===//
// Global Method Pool and Selector Serialization
//===----------------------------------------------------------------------===//

namespace {
// Trait used for the on-disk hash table used in the method pool.
class VISIBILITY_HIDDEN PCHMethodPoolTrait {
  PCHWriter &Writer;

public:
  typedef Selector key_type;
  typedef key_type key_type_ref;
  
  typedef std::pair<ObjCMethodList, ObjCMethodList> data_type;
  typedef const data_type& data_type_ref;

  explicit PCHMethodPoolTrait(PCHWriter &Writer) : Writer(Writer) { }
  
  static unsigned ComputeHash(Selector Sel) {
    unsigned N = Sel.getNumArgs();
    if (N == 0)
      ++N;
    unsigned R = 5381;
    for (unsigned I = 0; I != N; ++I)
      if (IdentifierInfo *II = Sel.getIdentifierInfoForSlot(I))
        R = clang::BernsteinHashPartial(II->getName(), II->getLength(), R);
    return R;
  }
  
  std::pair<unsigned,unsigned> 
    EmitKeyDataLength(llvm::raw_ostream& Out, Selector Sel,
                      data_type_ref Methods) {
    unsigned KeyLen = 2 + (Sel.getNumArgs()? Sel.getNumArgs() * 4 : 4);
    clang::io::Emit16(Out, KeyLen);
    unsigned DataLen = 2 + 2; // 2 bytes for each of the method counts
    for (const ObjCMethodList *Method = &Methods.first; Method; 
         Method = Method->Next)
      if (Method->Method)
        DataLen += 4;
    for (const ObjCMethodList *Method = &Methods.second; Method; 
         Method = Method->Next)
      if (Method->Method)
        DataLen += 4;
    clang::io::Emit16(Out, DataLen);
    return std::make_pair(KeyLen, DataLen);
  }
  
  void EmitKey(llvm::raw_ostream& Out, Selector Sel, unsigned) {
    uint64_t Start = Out.tell(); 
    assert((Start >> 32) == 0 && "Selector key offset too large");
    Writer.SetSelectorOffset(Sel, Start);
    unsigned N = Sel.getNumArgs();
    clang::io::Emit16(Out, N);
    if (N == 0)
      N = 1;
    for (unsigned I = 0; I != N; ++I)
      clang::io::Emit32(Out, 
                    Writer.getIdentifierRef(Sel.getIdentifierInfoForSlot(I)));
  }
  
  void EmitData(llvm::raw_ostream& Out, key_type_ref,
                data_type_ref Methods, unsigned DataLen) {
    uint64_t Start = Out.tell(); (void)Start;
    unsigned NumInstanceMethods = 0;
    for (const ObjCMethodList *Method = &Methods.first; Method; 
         Method = Method->Next)
      if (Method->Method)
        ++NumInstanceMethods;

    unsigned NumFactoryMethods = 0;
    for (const ObjCMethodList *Method = &Methods.second; Method; 
         Method = Method->Next)
      if (Method->Method)
        ++NumFactoryMethods;

    clang::io::Emit16(Out, NumInstanceMethods);
    clang::io::Emit16(Out, NumFactoryMethods);
    for (const ObjCMethodList *Method = &Methods.first; Method; 
         Method = Method->Next)
      if (Method->Method)
        clang::io::Emit32(Out, Writer.getDeclID(Method->Method));
    for (const ObjCMethodList *Method = &Methods.second; Method; 
         Method = Method->Next)
      if (Method->Method)
        clang::io::Emit32(Out, Writer.getDeclID(Method->Method));

    assert(Out.tell() - Start == DataLen && "Data length is wrong");
  }
};
} // end anonymous namespace

/// \brief Write the method pool into the PCH file.
///
/// The method pool contains both instance and factory methods, stored
/// in an on-disk hash table indexed by the selector.
void PCHWriter::WriteMethodPool(Sema &SemaRef) {
  using namespace llvm;

  // Create and write out the blob that contains the instance and
  // factor method pools.
  bool Empty = true;
  {
    OnDiskChainedHashTableGenerator<PCHMethodPoolTrait> Generator;
    
    // Create the on-disk hash table representation. Start by
    // iterating through the instance method pool.
    PCHMethodPoolTrait::key_type Key;
    unsigned NumSelectorsInMethodPool = 0;
    for (llvm::DenseMap<Selector, ObjCMethodList>::iterator
           Instance = SemaRef.InstanceMethodPool.begin(), 
           InstanceEnd = SemaRef.InstanceMethodPool.end();
         Instance != InstanceEnd; ++Instance) {
      // Check whether there is a factory method with the same
      // selector.
      llvm::DenseMap<Selector, ObjCMethodList>::iterator Factory
        = SemaRef.FactoryMethodPool.find(Instance->first);

      if (Factory == SemaRef.FactoryMethodPool.end())
        Generator.insert(Instance->first,
                         std::make_pair(Instance->second, 
                                        ObjCMethodList()));
      else
        Generator.insert(Instance->first,
                         std::make_pair(Instance->second, Factory->second));

      ++NumSelectorsInMethodPool;
      Empty = false;
    }

    // Now iterate through the factory method pool, to pick up any
    // selectors that weren't already in the instance method pool.
    for (llvm::DenseMap<Selector, ObjCMethodList>::iterator
           Factory = SemaRef.FactoryMethodPool.begin(), 
           FactoryEnd = SemaRef.FactoryMethodPool.end();
         Factory != FactoryEnd; ++Factory) {
      // Check whether there is an instance method with the same
      // selector. If so, there is no work to do here.
      llvm::DenseMap<Selector, ObjCMethodList>::iterator Instance
        = SemaRef.InstanceMethodPool.find(Factory->first);

      if (Instance == SemaRef.InstanceMethodPool.end()) {
        Generator.insert(Factory->first,
                         std::make_pair(ObjCMethodList(), Factory->second));
        ++NumSelectorsInMethodPool;
      }

      Empty = false;
    }

    if (Empty && SelectorOffsets.empty())
      return;

    // Create the on-disk hash table in a buffer.
    llvm::SmallVector<char, 4096> MethodPool; 
    uint32_t BucketOffset;
    SelectorOffsets.resize(SelVector.size());
    {
      PCHMethodPoolTrait Trait(*this);
      llvm::raw_svector_ostream Out(MethodPool);
      // Make sure that no bucket is at offset 0
      clang::io::Emit32(Out, 0);
      BucketOffset = Generator.Emit(Out, Trait);

      // For every selector that we have seen but which was not
      // written into the hash table, write the selector itself and
      // record it's offset.
      for (unsigned I = 0, N = SelVector.size(); I != N; ++I)
        if (SelectorOffsets[I] == 0)
          Trait.EmitKey(Out, SelVector[I], 0);
    }

    // Create a blob abbreviation
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(pch::METHOD_POOL));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned MethodPoolAbbrev = Stream.EmitAbbrev(Abbrev);

    // Write the method pool
    RecordData Record;
    Record.push_back(pch::METHOD_POOL);
    Record.push_back(BucketOffset);
    Record.push_back(NumSelectorsInMethodPool);
    Stream.EmitRecordWithBlob(MethodPoolAbbrev, Record, 
                              &MethodPool.front(), 
                              MethodPool.size());

    // Create a blob abbreviation for the selector table offsets.
    Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(pch::SELECTOR_OFFSETS));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // index
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned SelectorOffsetAbbrev = Stream.EmitAbbrev(Abbrev);

    // Write the selector offsets table.
    Record.clear();
    Record.push_back(pch::SELECTOR_OFFSETS);
    Record.push_back(SelectorOffsets.size());
    Stream.EmitRecordWithBlob(SelectorOffsetAbbrev, Record,
                              (const char *)&SelectorOffsets.front(),
                              SelectorOffsets.size() * 4);
  }
}

//===----------------------------------------------------------------------===//
// Identifier Table Serialization
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN PCHIdentifierTableTrait {
  PCHWriter &Writer;
  Preprocessor &PP;

  /// \brief Determines whether this is an "interesting" identifier
  /// that needs a full IdentifierInfo structure written into the hash
  /// table.
  static bool isInterestingIdentifier(const IdentifierInfo *II) {
    return II->isPoisoned() ||
      II->isExtensionToken() ||
      II->hasMacroDefinition() ||
      II->getObjCOrBuiltinID() ||
      II->getFETokenInfo<void>();
  }

public:
  typedef const IdentifierInfo* key_type;
  typedef key_type  key_type_ref;
  
  typedef pch::IdentID data_type;
  typedef data_type data_type_ref;
  
  PCHIdentifierTableTrait(PCHWriter &Writer, Preprocessor &PP) 
    : Writer(Writer), PP(PP) { }

  static unsigned ComputeHash(const IdentifierInfo* II) {
    return clang::BernsteinHash(II->getName());
  }
  
  std::pair<unsigned,unsigned> 
    EmitKeyDataLength(llvm::raw_ostream& Out, const IdentifierInfo* II, 
                      pch::IdentID ID) {
    unsigned KeyLen = strlen(II->getName()) + 1;
    unsigned DataLen = 4; // 4 bytes for the persistent ID << 1
    if (isInterestingIdentifier(II)) {
      DataLen += 2; // 2 bytes for builtin ID, flags
      if (II->hasMacroDefinition() && 
          !PP.getMacroInfo(const_cast<IdentifierInfo *>(II))->isBuiltinMacro())
        DataLen += 4;
      for (IdentifierResolver::iterator D = IdentifierResolver::begin(II),
                                     DEnd = IdentifierResolver::end();
           D != DEnd; ++D)
        DataLen += sizeof(pch::DeclID);
    }
    clang::io::Emit16(Out, DataLen);
    // We emit the key length after the data length so that every
    // string is preceded by a 16-bit length. This matches the PTH
    // format for storing identifiers.
    clang::io::Emit16(Out, KeyLen);
    return std::make_pair(KeyLen, DataLen);
  }
  
  void EmitKey(llvm::raw_ostream& Out, const IdentifierInfo* II, 
               unsigned KeyLen) {
    // Record the location of the key data.  This is used when generating
    // the mapping from persistent IDs to strings.
    Writer.SetIdentifierOffset(II, Out.tell());
    Out.write(II->getName(), KeyLen);
  }
  
  void EmitData(llvm::raw_ostream& Out, const IdentifierInfo* II, 
                pch::IdentID ID, unsigned) {
    if (!isInterestingIdentifier(II)) {
      clang::io::Emit32(Out, ID << 1);
      return;
    }

    clang::io::Emit32(Out, (ID << 1) | 0x01);
    uint32_t Bits = 0;
    bool hasMacroDefinition = 
      II->hasMacroDefinition() && 
      !PP.getMacroInfo(const_cast<IdentifierInfo *>(II))->isBuiltinMacro();
    Bits = (uint32_t)II->getObjCOrBuiltinID();
    Bits = (Bits << 1) | hasMacroDefinition;
    Bits = (Bits << 1) | II->isExtensionToken();
    Bits = (Bits << 1) | II->isPoisoned();
    Bits = (Bits << 1) | II->isCPlusPlusOperatorKeyword();
    clang::io::Emit16(Out, Bits);

    if (hasMacroDefinition)
      clang::io::Emit32(Out, Writer.getMacroOffset(II));

    // Emit the declaration IDs in reverse order, because the
    // IdentifierResolver provides the declarations as they would be
    // visible (e.g., the function "stat" would come before the struct
    // "stat"), but IdentifierResolver::AddDeclToIdentifierChain()
    // adds declarations to the end of the list (so we need to see the
    // struct "status" before the function "status").
    llvm::SmallVector<Decl *, 16> Decls(IdentifierResolver::begin(II), 
                                        IdentifierResolver::end());
    for (llvm::SmallVector<Decl *, 16>::reverse_iterator D = Decls.rbegin(),
                                                      DEnd = Decls.rend();
         D != DEnd; ++D)
      clang::io::Emit32(Out, Writer.getDeclID(*D));
  }
};
} // end anonymous namespace

/// \brief Write the identifier table into the PCH file.
///
/// The identifier table consists of a blob containing string data
/// (the actual identifiers themselves) and a separate "offsets" index
/// that maps identifier IDs to locations within the blob.
void PCHWriter::WriteIdentifierTable(Preprocessor &PP) {
  using namespace llvm;

  // Create and write out the blob that contains the identifier
  // strings.
  {
    OnDiskChainedHashTableGenerator<PCHIdentifierTableTrait> Generator;
    
    // Look for any identifiers that were named while processing the
    // headers, but are otherwise not needed. We add these to the hash
    // table to enable checking of the predefines buffer in the case
    // where the user adds new macro definitions when building the PCH
    // file.
    for (IdentifierTable::iterator ID = PP.getIdentifierTable().begin(),
                                IDEnd = PP.getIdentifierTable().end();
         ID != IDEnd; ++ID)
      getIdentifierRef(ID->second);

    // Create the on-disk hash table representation.
    IdentifierOffsets.resize(IdentifierIDs.size());
    for (llvm::DenseMap<const IdentifierInfo *, pch::IdentID>::iterator
           ID = IdentifierIDs.begin(), IDEnd = IdentifierIDs.end();
         ID != IDEnd; ++ID) {
      assert(ID->first && "NULL identifier in identifier table");
      Generator.insert(ID->first, ID->second);
    }

    // Create the on-disk hash table in a buffer.
    llvm::SmallVector<char, 4096> IdentifierTable; 
    uint32_t BucketOffset;
    {
      PCHIdentifierTableTrait Trait(*this, PP);
      llvm::raw_svector_ostream Out(IdentifierTable);
      // Make sure that no bucket is at offset 0
      clang::io::Emit32(Out, 0);
      BucketOffset = Generator.Emit(Out, Trait);
    }

    // Create a blob abbreviation
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(pch::IDENTIFIER_TABLE));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned IDTableAbbrev = Stream.EmitAbbrev(Abbrev);

    // Write the identifier table
    RecordData Record;
    Record.push_back(pch::IDENTIFIER_TABLE);
    Record.push_back(BucketOffset);
    Stream.EmitRecordWithBlob(IDTableAbbrev, Record, 
                              &IdentifierTable.front(), 
                              IdentifierTable.size());
  }

  // Write the offsets table for identifier IDs.
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::IDENTIFIER_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of identifiers
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned IdentifierOffsetAbbrev = Stream.EmitAbbrev(Abbrev);

  RecordData Record;
  Record.push_back(pch::IDENTIFIER_OFFSET);
  Record.push_back(IdentifierOffsets.size());
  Stream.EmitRecordWithBlob(IdentifierOffsetAbbrev, Record,
                            (const char *)&IdentifierOffsets.front(),
                            IdentifierOffsets.size() * sizeof(uint32_t));
}

//===----------------------------------------------------------------------===//
// General Serialization Routines
//===----------------------------------------------------------------------===//

/// \brief Write a record containing the given attributes.
void PCHWriter::WriteAttributeRecord(const Attr *Attr) {
  RecordData Record;
  for (; Attr; Attr = Attr->getNext()) {
    Record.push_back(Attr->getKind()); // FIXME: stable encoding
    Record.push_back(Attr->isInherited());
    switch (Attr->getKind()) {
    case Attr::Alias:
      AddString(cast<AliasAttr>(Attr)->getAliasee(), Record);
      break;

    case Attr::Aligned:
      Record.push_back(cast<AlignedAttr>(Attr)->getAlignment());
      break;

    case Attr::AlwaysInline:
      break;
     
    case Attr::AnalyzerNoReturn:
      break;

    case Attr::Annotate:
      AddString(cast<AnnotateAttr>(Attr)->getAnnotation(), Record);
      break;

    case Attr::AsmLabel:
      AddString(cast<AsmLabelAttr>(Attr)->getLabel(), Record);
      break;

    case Attr::Blocks:
      Record.push_back(cast<BlocksAttr>(Attr)->getType()); // FIXME: stable
      break;

    case Attr::Cleanup:
      AddDeclRef(cast<CleanupAttr>(Attr)->getFunctionDecl(), Record);
      break;

    case Attr::Const:
      break;

    case Attr::Constructor:
      Record.push_back(cast<ConstructorAttr>(Attr)->getPriority());
      break;

    case Attr::DLLExport:
    case Attr::DLLImport:
    case Attr::Deprecated:
      break;

    case Attr::Destructor:
      Record.push_back(cast<DestructorAttr>(Attr)->getPriority());
      break;

    case Attr::FastCall:
      break;

    case Attr::Format: {
      const FormatAttr *Format = cast<FormatAttr>(Attr);
      AddString(Format->getType(), Record);
      Record.push_back(Format->getFormatIdx());
      Record.push_back(Format->getFirstArg());
      break;
    }

    case Attr::FormatArg: {
      const FormatArgAttr *Format = cast<FormatArgAttr>(Attr);
      Record.push_back(Format->getFormatIdx());
      break;
    }

    case Attr::Sentinel : {
      const SentinelAttr *Sentinel = cast<SentinelAttr>(Attr);
      Record.push_back(Sentinel->getSentinel());
      Record.push_back(Sentinel->getNullPos());
      break;
    }
        
    case Attr::GNUInline:
    case Attr::IBOutletKind:
    case Attr::Malloc:
    case Attr::NoReturn:
    case Attr::NoThrow:
    case Attr::Nodebug:
    case Attr::Noinline:
      break;

    case Attr::NonNull: {
      const NonNullAttr *NonNull = cast<NonNullAttr>(Attr);
      Record.push_back(NonNull->size());
      Record.insert(Record.end(), NonNull->begin(), NonNull->end());
      break;
    }

    case Attr::ObjCException:
    case Attr::ObjCNSObject:
    case Attr::CFReturnsRetained:
    case Attr::NSReturnsRetained:
    case Attr::Overloadable:
      break;

    case Attr::PragmaPack:
      Record.push_back(cast<PragmaPackAttr>(Attr)->getAlignment());
      break;

    case Attr::Packed:
      break;
    
    case Attr::Pure:
      break;

    case Attr::Regparm:
      Record.push_back(cast<RegparmAttr>(Attr)->getNumParams());
      break;
        
    case Attr::ReqdWorkGroupSize:
      Record.push_back(cast<ReqdWorkGroupSizeAttr>(Attr)->getXDim());
      Record.push_back(cast<ReqdWorkGroupSizeAttr>(Attr)->getYDim());
      Record.push_back(cast<ReqdWorkGroupSizeAttr>(Attr)->getZDim());
      break;

    case Attr::Section:
      AddString(cast<SectionAttr>(Attr)->getName(), Record);
      break;

    case Attr::StdCall:
    case Attr::TransparentUnion:
    case Attr::Unavailable:
    case Attr::Unused:
    case Attr::Used:
      break;

    case Attr::Visibility:
      // FIXME: stable encoding
      Record.push_back(cast<VisibilityAttr>(Attr)->getVisibility()); 
      break;

    case Attr::WarnUnusedResult:
    case Attr::Weak:
    case Attr::WeakImport:
      break;
    }
  }

  Stream.EmitRecord(pch::DECL_ATTR, Record);
}

void PCHWriter::AddString(const std::string &Str, RecordData &Record) {
  Record.push_back(Str.size());
  Record.insert(Record.end(), Str.begin(), Str.end());
}

/// \brief Note that the identifier II occurs at the given offset
/// within the identifier table.
void PCHWriter::SetIdentifierOffset(const IdentifierInfo *II, uint32_t Offset) {
  IdentifierOffsets[IdentifierIDs[II] - 1] = Offset;
}

/// \brief Note that the selector Sel occurs at the given offset
/// within the method pool/selector table.
void PCHWriter::SetSelectorOffset(Selector Sel, uint32_t Offset) {
  unsigned ID = SelectorIDs[Sel];
  assert(ID && "Unknown selector");
  SelectorOffsets[ID - 1] = Offset;
}

PCHWriter::PCHWriter(llvm::BitstreamWriter &Stream) 
  : Stream(Stream), NextTypeID(pch::NUM_PREDEF_TYPE_IDS), 
    NumStatements(0), NumMacros(0), NumLexicalDeclContexts(0),
    NumVisibleDeclContexts(0) { }

void PCHWriter::WritePCH(Sema &SemaRef, MemorizeStatCalls *StatCalls,
                         const char *isysroot) {
  using namespace llvm;

  ASTContext &Context = SemaRef.Context;
  Preprocessor &PP = SemaRef.PP;

  // Emit the file header.
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'P', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'H', 8);
  
  WriteBlockInfoBlock();

  // The translation unit is the first declaration we'll emit.
  DeclIDs[Context.getTranslationUnitDecl()] = 1;
  DeclsToEmit.push(Context.getTranslationUnitDecl());

  // Make sure that we emit IdentifierInfos (and any attached
  // declarations) for builtins.
  {
    IdentifierTable &Table = PP.getIdentifierTable();
    llvm::SmallVector<const char *, 32> BuiltinNames;
    Context.BuiltinInfo.GetBuiltinNames(BuiltinNames,
                                        Context.getLangOptions().NoBuiltin);
    for (unsigned I = 0, N = BuiltinNames.size(); I != N; ++I)
      getIdentifierRef(&Table.get(BuiltinNames[I]));
  }

  // Build a record containing all of the tentative definitions in this file, in
  // TentativeDefinitionList order.  Generally, this record will be empty for
  // headers.
  RecordData TentativeDefinitions;
  for (unsigned i = 0, e = SemaRef.TentativeDefinitionList.size(); i != e; ++i){
    VarDecl *VD =
      SemaRef.TentativeDefinitions.lookup(SemaRef.TentativeDefinitionList[i]);
    if (VD) AddDeclRef(VD, TentativeDefinitions);
  }

  // Build a record containing all of the locally-scoped external
  // declarations in this header file. Generally, this record will be
  // empty.
  RecordData LocallyScopedExternalDecls;
  // FIXME: This is filling in the PCH file in densemap order which is
  // nondeterminstic!
  for (llvm::DenseMap<DeclarationName, NamedDecl *>::iterator 
         TD = SemaRef.LocallyScopedExternalDecls.begin(),
         TDEnd = SemaRef.LocallyScopedExternalDecls.end();
       TD != TDEnd; ++TD)
    AddDeclRef(TD->second, LocallyScopedExternalDecls);

  // Build a record containing all of the ext_vector declarations.
  RecordData ExtVectorDecls;
  for (unsigned I = 0, N = SemaRef.ExtVectorDecls.size(); I != N; ++I)
    AddDeclRef(SemaRef.ExtVectorDecls[I], ExtVectorDecls);

  // Write the remaining PCH contents.
  RecordData Record;
  Stream.EnterSubblock(pch::PCH_BLOCK_ID, 4);
  WriteMetadata(Context, isysroot);
  WriteLanguageOptions(Context.getLangOptions());
  if (StatCalls && !isysroot)
    WriteStatCache(*StatCalls, isysroot);
  WriteSourceManagerBlock(Context.getSourceManager(), PP, isysroot);
  WritePreprocessor(PP);
  WriteComments(Context);  
  // Write the record of special types.
  Record.clear();
  
  AddTypeRef(Context.getBuiltinVaListType(), Record);
  AddTypeRef(Context.getObjCIdType(), Record);
  AddTypeRef(Context.getObjCSelType(), Record);
  AddTypeRef(Context.getObjCProtoType(), Record);
  AddTypeRef(Context.getObjCClassType(), Record);
  AddTypeRef(Context.getRawCFConstantStringType(), Record);
  AddTypeRef(Context.getRawObjCFastEnumerationStateType(), Record);
  AddTypeRef(Context.getFILEType(), Record);
  AddTypeRef(Context.getjmp_bufType(), Record);
  AddTypeRef(Context.getsigjmp_bufType(), Record);
  AddTypeRef(Context.ObjCIdRedefinitionType, Record);
  AddTypeRef(Context.ObjCClassRedefinitionType, Record);
  Stream.EmitRecord(pch::SPECIAL_TYPES, Record);
  
  // Keep writing types and declarations until all types and
  // declarations have been written.
  do {
    if (!DeclsToEmit.empty())
      WriteDeclsBlock(Context);
    if (!TypesToEmit.empty())
      WriteTypesBlock(Context);
  } while (!(DeclsToEmit.empty() && TypesToEmit.empty()));

  WriteMethodPool(SemaRef);
  WriteIdentifierTable(PP);

  // Write the type offsets array
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::TYPE_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of types
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // types block
  unsigned TypeOffsetAbbrev = Stream.EmitAbbrev(Abbrev);
  Record.clear();
  Record.push_back(pch::TYPE_OFFSET);
  Record.push_back(TypeOffsets.size());
  Stream.EmitRecordWithBlob(TypeOffsetAbbrev, Record,
                            (const char *)&TypeOffsets.front(), 
                            TypeOffsets.size() * sizeof(TypeOffsets[0]));
  
  // Write the declaration offsets array
  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::DECL_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of declarations
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // declarations block
  unsigned DeclOffsetAbbrev = Stream.EmitAbbrev(Abbrev);
  Record.clear();
  Record.push_back(pch::DECL_OFFSET);
  Record.push_back(DeclOffsets.size());
  Stream.EmitRecordWithBlob(DeclOffsetAbbrev, Record,
                            (const char *)&DeclOffsets.front(), 
                            DeclOffsets.size() * sizeof(DeclOffsets[0]));

  // Write the record containing external, unnamed definitions.
  if (!ExternalDefinitions.empty())
    Stream.EmitRecord(pch::EXTERNAL_DEFINITIONS, ExternalDefinitions);

  // Write the record containing tentative definitions.
  if (!TentativeDefinitions.empty())
    Stream.EmitRecord(pch::TENTATIVE_DEFINITIONS, TentativeDefinitions);

  // Write the record containing locally-scoped external definitions.
  if (!LocallyScopedExternalDecls.empty())
    Stream.EmitRecord(pch::LOCALLY_SCOPED_EXTERNAL_DECLS, 
                      LocallyScopedExternalDecls);

  // Write the record containing ext_vector type names.
  if (!ExtVectorDecls.empty())
    Stream.EmitRecord(pch::EXT_VECTOR_DECLS, ExtVectorDecls);
  
  // Some simple statistics
  Record.clear();
  Record.push_back(NumStatements);
  Record.push_back(NumMacros);
  Record.push_back(NumLexicalDeclContexts);
  Record.push_back(NumVisibleDeclContexts);
  Stream.EmitRecord(pch::STATISTICS, Record);
  Stream.ExitBlock();
}

void PCHWriter::AddSourceLocation(SourceLocation Loc, RecordData &Record) {
  Record.push_back(Loc.getRawEncoding());
}

void PCHWriter::AddAPInt(const llvm::APInt &Value, RecordData &Record) {
  Record.push_back(Value.getBitWidth());
  unsigned N = Value.getNumWords();
  const uint64_t* Words = Value.getRawData();
  for (unsigned I = 0; I != N; ++I)
    Record.push_back(Words[I]);
}

void PCHWriter::AddAPSInt(const llvm::APSInt &Value, RecordData &Record) {
  Record.push_back(Value.isUnsigned());
  AddAPInt(Value, Record);
}

void PCHWriter::AddAPFloat(const llvm::APFloat &Value, RecordData &Record) {
  AddAPInt(Value.bitcastToAPInt(), Record);
}

void PCHWriter::AddIdentifierRef(const IdentifierInfo *II, RecordData &Record) {
  Record.push_back(getIdentifierRef(II));
}

pch::IdentID PCHWriter::getIdentifierRef(const IdentifierInfo *II) {
  if (II == 0)
    return 0;

  pch::IdentID &ID = IdentifierIDs[II];
  if (ID == 0)
    ID = IdentifierIDs.size();
  return ID;
}

void PCHWriter::AddSelectorRef(const Selector SelRef, RecordData &Record) {
  if (SelRef.getAsOpaquePtr() == 0) {
    Record.push_back(0);
    return;
  }

  pch::SelectorID &SID = SelectorIDs[SelRef];
  if (SID == 0) {
    SID = SelectorIDs.size();
    SelVector.push_back(SelRef);
  }
  Record.push_back(SID);
}

void PCHWriter::AddTypeRef(QualType T, RecordData &Record) {
  if (T.isNull()) {
    Record.push_back(pch::PREDEF_TYPE_NULL_ID);
    return;
  }

  if (const BuiltinType *BT = dyn_cast<BuiltinType>(T.getTypePtr())) {
    pch::TypeID ID = 0;
    switch (BT->getKind()) {
    case BuiltinType::Void:       ID = pch::PREDEF_TYPE_VOID_ID;       break;
    case BuiltinType::Bool:       ID = pch::PREDEF_TYPE_BOOL_ID;       break;
    case BuiltinType::Char_U:     ID = pch::PREDEF_TYPE_CHAR_U_ID;     break;
    case BuiltinType::UChar:      ID = pch::PREDEF_TYPE_UCHAR_ID;      break;
    case BuiltinType::UShort:     ID = pch::PREDEF_TYPE_USHORT_ID;     break;
    case BuiltinType::UInt:       ID = pch::PREDEF_TYPE_UINT_ID;       break;
    case BuiltinType::ULong:      ID = pch::PREDEF_TYPE_ULONG_ID;      break;
    case BuiltinType::ULongLong:  ID = pch::PREDEF_TYPE_ULONGLONG_ID;  break;
    case BuiltinType::UInt128:    ID = pch::PREDEF_TYPE_UINT128_ID;    break;
    case BuiltinType::Char_S:     ID = pch::PREDEF_TYPE_CHAR_S_ID;     break;
    case BuiltinType::SChar:      ID = pch::PREDEF_TYPE_SCHAR_ID;      break;
    case BuiltinType::WChar:      ID = pch::PREDEF_TYPE_WCHAR_ID;      break;
    case BuiltinType::Short:      ID = pch::PREDEF_TYPE_SHORT_ID;      break;
    case BuiltinType::Int:        ID = pch::PREDEF_TYPE_INT_ID;        break;
    case BuiltinType::Long:       ID = pch::PREDEF_TYPE_LONG_ID;       break;
    case BuiltinType::LongLong:   ID = pch::PREDEF_TYPE_LONGLONG_ID;   break;
    case BuiltinType::Int128:     ID = pch::PREDEF_TYPE_INT128_ID;     break;
    case BuiltinType::Float:      ID = pch::PREDEF_TYPE_FLOAT_ID;      break;
    case BuiltinType::Double:     ID = pch::PREDEF_TYPE_DOUBLE_ID;     break;
    case BuiltinType::LongDouble: ID = pch::PREDEF_TYPE_LONGDOUBLE_ID; break;
    case BuiltinType::NullPtr:    ID = pch::PREDEF_TYPE_NULLPTR_ID;    break;
    case BuiltinType::Char16:     ID = pch::PREDEF_TYPE_CHAR16_ID;     break;
    case BuiltinType::Char32:     ID = pch::PREDEF_TYPE_CHAR32_ID;     break;
    case BuiltinType::Overload:   ID = pch::PREDEF_TYPE_OVERLOAD_ID;   break;
    case BuiltinType::Dependent:  ID = pch::PREDEF_TYPE_DEPENDENT_ID;  break;
    case BuiltinType::ObjCId:     ID = pch::PREDEF_TYPE_OBJC_ID;       break;
    case BuiltinType::ObjCClass:  ID = pch::PREDEF_TYPE_OBJC_CLASS;    break;
    case BuiltinType::UndeducedAuto:
      assert(0 && "Should not see undeduced auto here");
      break;
    }

    Record.push_back((ID << 3) | T.getCVRQualifiers());
    return;
  }

  pch::TypeID &ID = TypeIDs[T.getTypePtr()];
  if (ID == 0) {
    // We haven't seen this type before. Assign it a new ID and put it
    // into the queu of types to emit.
    ID = NextTypeID++;
    TypesToEmit.push(T.getTypePtr());
  }

  // Encode the type qualifiers in the type reference.
  Record.push_back((ID << 3) | T.getCVRQualifiers());
}

void PCHWriter::AddDeclRef(const Decl *D, RecordData &Record) {
  if (D == 0) {
    Record.push_back(0);
    return;
  }

  pch::DeclID &ID = DeclIDs[D];
  if (ID == 0) { 
    // We haven't seen this declaration before. Give it a new ID and
    // enqueue it in the list of declarations to emit.
    ID = DeclIDs.size();
    DeclsToEmit.push(const_cast<Decl *>(D));
  }

  Record.push_back(ID);
}

pch::DeclID PCHWriter::getDeclID(const Decl *D) {
  if (D == 0)
    return 0;

  assert(DeclIDs.find(D) != DeclIDs.end() && "Declaration not emitted!");
  return DeclIDs[D];
}

void PCHWriter::AddDeclarationName(DeclarationName Name, RecordData &Record) {
  // FIXME: Emit a stable enum for NameKind.  0 = Identifier etc.
  Record.push_back(Name.getNameKind());
  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
    AddIdentifierRef(Name.getAsIdentifierInfo(), Record);
    break;

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    AddSelectorRef(Name.getObjCSelector(), Record);
    break;

  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    AddTypeRef(Name.getCXXNameType(), Record);
    break;

  case DeclarationName::CXXOperatorName:
    Record.push_back(Name.getCXXOverloadedOperator());
    break;

  case DeclarationName::CXXUsingDirective:
    // No extra data to emit
    break;
  }
}

