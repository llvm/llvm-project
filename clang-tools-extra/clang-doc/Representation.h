///===-- Representation.h - ClangDoc Representation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the internal representations of different declaration
// types for the clang-doc tool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_REPRESENTATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_REPRESENTATION_H

#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Tooling/Execution.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/simple_ilist.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/StringSaver.h"
#include <array>
#include <memory>
#include <optional>
#include <string>

namespace clang {
namespace doc {

class ConcurrentStringPool {
public:
  StringRef intern(StringRef Name) {
    if (Name.empty())
      return StringRef();

    llvm::sys::SmartScopedLock<true> Lock(PoolMutex);
    return Saver.save(Name);
  }

private:
  llvm::sys::SmartMutex<true> PoolMutex;
  llvm::BumpPtrAllocator Alloc;
  llvm::UniqueStringSaver Saver{Alloc};
};

ConcurrentStringPool &getGlobalStringPool();

extern thread_local llvm::BumpPtrAllocator TransientArena;

inline StringRef internString(const Twine &T) {
  if (T.isTriviallyEmpty())
    return StringRef();

  if (T.isSingleStringRef()) {
    StringRef S = T.getSingleStringRef();
    if (S.empty())
      return StringRef();
    return getGlobalStringPool().intern(S);
  }

  SmallString<128> Buffer;
  StringRef S = T.toStringRef(Buffer);
  if (S.empty())
    return StringRef();
  return getGlobalStringPool().intern(S);
}

template <typename T>
inline llvm::ArrayRef<T> allocateArray(llvm::ArrayRef<T> V,
                                       llvm::BumpPtrAllocator &Alloc) {
  if (V.empty())
    return llvm::ArrayRef<T>();
  T *Allocated = (T *)Alloc.Allocate<T>(V.size());
  std::uninitialized_move(V.begin(), V.end(), Allocated);
  return llvm::ArrayRef<T>(Allocated, V.size());
}

// An abstraction for owned pointers. Initially mapped to OwnedPtr,
// to be eventually transitioned to bare pointers in an arena.
template <typename T> using OwnedPtr = std::unique_ptr<T>;

// An abstraction for vectors that are populated and read sequentially.
// To be eventually transitioned to llvm::ArrayRef for arena storage.
template <typename T> using OwningArray = std::vector<T>;

// An abstraction for lists that are dynamically managed (inserted/removed).
// To be eventually transitioned to llvm::simple_ilist.
template <typename T> using OwningVec = std::vector<T>;

// An abstraction for dynamic lists of owned pointers.
// To be eventually transitioned to llvm::simple_ilist<T*> or similar.
template <typename T> using OwningPtrVec = std::vector<OwnedPtr<T>>;

// An abstraction for arrays of owned pointers.
// To be eventually transitioned to arena-allocated arrays of bare pointers.
template <typename T> using OwningPtrArray = std::vector<OwnedPtr<T>>;

// A helper function to create an owned pointer, abstracting away the memory
// allocation mechanism.
template <typename T, typename... Args>
OwnedPtr<T> allocatePtr(Args &&...args) {
  return std::make_unique<T>(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
T *allocatePtr(llvm::BumpPtrAllocator &Alloc, Args &&...args) {
  return new (Alloc.Allocate<T>()) T(std::forward<Args>(args)...);
}

// A helper function to access the underlying pointer from an owned pointer,
// abstracting away the pointer dereferencing mechanism.
template <typename T> T *getPtr(const OwnedPtr<T> &O) { return O.get(); }

// SHA1'd hash of a USR.
using SymbolID = std::array<uint8_t, 20>;

constexpr SymbolID GlobalNamespaceID = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

struct BaseRecordInfo;
struct EnumInfo;
struct FunctionInfo;
struct Info;
struct TypedefInfo;
struct ConceptInfo;
struct VarInfo;

enum class InfoType {
  IT_default,
  IT_namespace,
  IT_record,
  IT_function,
  IT_enum,
  IT_typedef,
  IT_concept,
  IT_variable,
  IT_friend
};

enum class CommentKind {
  CK_FullComment,
  CK_ParagraphComment,
  CK_TextComment,
  CK_InlineCommandComment,
  CK_HTMLStartTagComment,
  CK_HTMLEndTagComment,
  CK_BlockCommandComment,
  CK_ParamCommandComment,
  CK_TParamCommandComment,
  CK_VerbatimBlockComment,
  CK_VerbatimBlockLineComment,
  CK_VerbatimLineComment,
  CK_Unknown
};

enum OutputFormatTy { md, yaml, html, json, md_mustache };

CommentKind stringToCommentKind(llvm::StringRef KindStr);
llvm::StringRef commentKindToString(CommentKind Kind);

// A representation of a parsed comment.
struct CommentInfo : public llvm::ilist_node<CommentInfo> {
  CommentInfo() = default;
  CommentInfo(CommentInfo &Other) = delete;
  CommentInfo(CommentInfo &&Other) = default;
  CommentInfo &operator=(CommentInfo &&Other) = default;

  bool operator==(const CommentInfo &Other) const;

  // This operator is used to sort a vector of CommentInfos.
  // No specific order (attributes more important than others) is required. Any
  // sort is enough, the order is only needed to call std::unique after sorting
  // the vector.
  bool operator<(const CommentInfo &Other) const;

  OwningPtrVec<CommentInfo>
      Children;              // List of child comments for this CommentInfo.
  StringRef Direction;       // Parameter direction (for (T)ParamCommand).
  StringRef Name;            // Name of the comment (for Verbatim and HTML).
  StringRef ParamName;       // Parameter name (for (T)ParamCommand).
  StringRef CloseName;       // Closing tag name (for VerbatimBlock).
  StringRef Text;            // Text of the comment.
  llvm::ArrayRef<StringRef> AttrKeys; // List of attribute keys (for HTML).
  llvm::ArrayRef<StringRef>
      AttrValues; // List of attribute values for each key (for HTML).
  llvm::ArrayRef<StringRef>
      Args; // List of arguments to commands (for InlineCommand).
  CommentKind Kind = CommentKind::
      CK_Unknown; // Kind of comment (FullComment, ParagraphComment,
                  // TextComment, InlineCommandComment, HTMLStartTagComment,
                  // HTMLEndTagComment, BlockCommandComment,
                  // ParamCommandComment, TParamCommandComment,
                  // VerbatimBlockComment, VerbatimBlockLineComment,
                  // VerbatimLineComment).
  bool SelfClosing = false; // Indicates if tag is self-closing (for HTML).
  bool Explicit = false;    // Indicates if the direction of a param is explicit
                            // (for (T)ParamCommand).
};

struct Reference : public llvm::ilist_node<Reference> {
  // This variant (that takes no qualified name parameter) uses the Name as the
  // QualName (very useful in unit tests to reduce verbosity). This can't use an
  // empty string to indicate the default because we need to accept the empty
  // string as a valid input for the global namespace (it will have
  // "GlobalNamespace" as the name, but an empty QualName).
  Reference(SymbolID USR = SymbolID(), StringRef Name = StringRef(),
            InfoType IT = InfoType::IT_default)
      : USR(USR), RefType(IT), Name(internString(Name)),
        QualName(internString(Name)) {}
  Reference(SymbolID USR, StringRef Name, InfoType IT, StringRef QualName,
            StringRef Path = StringRef())
      : USR(USR), RefType(IT), Name(internString(Name)),
        QualName(internString(QualName)), Path(internString(Path)) {}
  Reference(SymbolID USR, StringRef Name, InfoType IT, StringRef QualName,
            StringRef Path, StringRef DocumentationFileName)
      : USR(USR), RefType(IT), Name(internString(Name)),
        QualName(internString(QualName)), Path(internString(Path)),
        DocumentationFileName(internString(DocumentationFileName)) {}

  bool operator==(const Reference &Other) const {
    return std::tie(USR, Name, QualName, RefType) ==
           std::tie(Other.USR, Other.Name, QualName, Other.RefType);
  }

  bool mergeable(const Reference &Other);
  void merge(Reference &&I);
  bool operator<(const Reference &Other) const { return Name < Other.Name; }

  /// Returns the path for this Reference relative to CurrentPath.
  StringRef getRelativeFilePath(const StringRef &CurrentPath) const;

  /// Returns the basename that should be used for this Reference.
  StringRef getFileBaseName() const;

  SymbolID USR = SymbolID(); // Unique identifier for referenced decl

  InfoType RefType = InfoType::IT_default; // Indicates the type of this
                                           // Reference (namespace, record,
                                           // function, enum, default).

  // Name of type (possibly unresolved). Not including namespaces or template
  // parameters (so for a std::vector<int> this would be "vector"). See also
  // QualName.
  StringRef Name;

  // Full qualified name of this type, including namespaces and template
  // parameter (for example this could be "std::vector<int>"). Contrast to
  // Name.
  StringRef QualName;

  // Path of directory where the clang-doc generated file will be saved
  // (possibly unresolved)
  StringRef Path;
  StringRef DocumentationFileName;
};

// A Context is a reference that holds a relative path from a certain Info's
// location.
struct Context : public Reference {
  Context(SymbolID USR, StringRef Name, InfoType IT, StringRef QualName,
          StringRef Path, StringRef DocumentationFileName)
      : Reference(USR, Name, IT, QualName, Path, DocumentationFileName) {}
  explicit Context(const Info &I);
  StringRef RelativePath;
};

// Holds the children of a record or namespace.
struct ScopeChildren {
  // Namespaces and Records are references because they will be properly
  // documented in their own info, while the entirety of Functions and Enums are
  // included here because they should not have separate documentation from
  // their scope.
  //
  // Namespaces are not syntactically valid as children of records, but making
  // this general for all possible container types reduces code complexity.
  llvm::simple_ilist<Reference> Namespaces;
  OwningVec<Reference> Records;
  OwningVec<FunctionInfo> Functions;
  OwningVec<EnumInfo> Enums;
  OwningVec<TypedefInfo> Typedefs;
  OwningVec<ConceptInfo> Concepts;
  OwningVec<VarInfo> Variables;

  void sort();
};

// A base struct for TypeInfos
struct TypeInfo {
  TypeInfo() = default;
  TypeInfo(const Reference &R) : Type(R) {}

  // Convenience constructor for when there is no symbol ID or info type
  // (normally used for built-in types in tests).
  TypeInfo(StringRef Name, StringRef Path = StringRef())
      : Type(SymbolID(), Name, InfoType::IT_default, Name, Path) {}

  bool operator==(const TypeInfo &Other) const { return Type == Other.Type; }

  Reference Type; // Referenced type in this info.

  bool IsTemplate = false;
  bool IsBuiltIn = false;
};

// Represents one template parameter.
//
// This is a very simple serialization of the text of the source code of the
// template parameter. It is saved in a struct so there is a place to add the
// name and default values in the future if needed.
struct TemplateParamInfo {
  TemplateParamInfo() = default;
  explicit TemplateParamInfo(StringRef Contents)
      : Contents(internString(Contents)) {}

  // The literal contents of the code for that specifies this template parameter
  // for this declaration. Typical values will be "class T" and
  // "typename T = int".
  StringRef Contents;
};

struct TemplateSpecializationInfo {
  // Indicates the declaration that this specializes.
  SymbolID SpecializationOf;

  // Template parameters applying to the specialized record/function.
  OwningVec<TemplateParamInfo> Params;
};

struct ConstraintInfo {
  ConstraintInfo() = default;
  ConstraintInfo(SymbolID USR, StringRef Name)
      : ConceptRef(USR, Name, InfoType::IT_concept) {}
  Reference ConceptRef;

  StringRef ConstraintExpr;
};

// Records the template information for a struct or function that is a template
// or an explicit template specialization.
struct TemplateInfo {
  // May be empty for non-partial specializations.
  OwningVec<TemplateParamInfo> Params;

  // Set when this is a specialization of another record/function.
  std::optional<TemplateSpecializationInfo> Specialization;
  OwningVec<ConstraintInfo> Constraints;
};

// Info for field types.
struct FieldTypeInfo : public TypeInfo {
  FieldTypeInfo() = default;
  FieldTypeInfo(const TypeInfo &TI, StringRef Name = StringRef(),
                StringRef DefaultValue = StringRef())
      : TypeInfo(TI), Name(internString(Name)),
        DefaultValue(internString(DefaultValue)) {}

  bool operator==(const FieldTypeInfo &Other) const {
    return std::tie(Type, Name, DefaultValue) ==
           std::tie(Other.Type, Other.Name, Other.DefaultValue);
  }

  StringRef Name; // Name associated with this info.

  // When used for function parameters, contains the string representing the
  // expression of the default value, if any.
  StringRef DefaultValue;
};

// Info for member types.
struct MemberTypeInfo : public FieldTypeInfo {
  MemberTypeInfo() = default;
  MemberTypeInfo(const TypeInfo &TI, StringRef Name, AccessSpecifier Access,
                 bool IsStatic = false)
      : FieldTypeInfo(TI, Name), Access(Access), IsStatic(IsStatic) {}

  bool operator==(const MemberTypeInfo &Other) const {
    return std::tie(Type, Name, Access, IsStatic, Description) ==
           std::tie(Other.Type, Other.Name, Other.Access, Other.IsStatic,
                    Other.Description);
  }

  OwningVec<CommentInfo> Description;

  // Access level associated with this info (public, protected, private, none).
  // AS_public is set as default because the bitcode writer requires the enum
  // with value 0 to be used as the default.
  // (AS_public = 0, AS_protected = 1, AS_private = 2, AS_none = 3)
  AccessSpecifier Access = AccessSpecifier::AS_public;
  bool IsStatic = false;
};

struct Location : public llvm::ilist_node<Location> {
  Location(int StartLineNumber = 0, int EndLineNumber = 0,
           StringRef Filename = StringRef(), bool IsFileInRootDir = false)
      : Filename(internString(Filename)), StartLineNumber(StartLineNumber),
        EndLineNumber(EndLineNumber), IsFileInRootDir(IsFileInRootDir) {}

  bool operator==(const Location &Other) const {
    return std::tie(StartLineNumber, EndLineNumber, Filename) ==
           std::tie(Other.StartLineNumber, Other.EndLineNumber, Other.Filename);
  }

  bool operator!=(const Location &Other) const { return !(*this == Other); }

  // This operator is used to sort a vector of Locations.
  // No specific order (attributes more important than others) is required. Any
  // sort is enough, the order is only needed to call std::unique after sorting
  // the vector.
  bool operator<(const Location &Other) const {
    return std::tie(StartLineNumber, EndLineNumber, Filename) <
           std::tie(Other.StartLineNumber, Other.EndLineNumber, Other.Filename);
  }

  StringRef Filename;
  int StartLineNumber = 0;
  int EndLineNumber = 0;
  bool IsFileInRootDir = false;
};

/// A base struct for Infos.
struct Info {
  Info(InfoType IT = InfoType::IT_default, SymbolID USR = SymbolID(),
       StringRef Name = StringRef(), StringRef Path = StringRef())
      : Path(internString(Path)), Name(internString(Name)), USR(USR), IT(IT) {}

  Info(const Info &Other) = delete;
  Info(Info &&Other) = default;
  virtual ~Info() = default;

  Info &operator=(Info &&Other) = default;

  void mergeBase(Info &&I);
  bool mergeable(const Info &Other);

  StringRef extractName() const;

  /// Returns the file path for this Info relative to CurrentPath.
  StringRef getRelativeFilePath(const StringRef &CurrentPath) const;

  /// Returns the basename that should be used for this Info.
  StringRef getFileBaseName() const;

  // Path of directory where the clang-doc generated file will be saved.
  StringRef Path;

  // Unqualified name of the decl.
  StringRef Name;

  // The name used for the file that this info is documented in.
  // In the JSON generator, infos are documented in files with mangled names.
  // Thus, we keep track of the physical filename for linking purposes.
  StringRef DocumentationFileName;

  // List of parent namespaces for this decl.
  llvm::SmallVector<Reference, 4> Namespace;

  // Unique identifier for the decl described by this Info.
  SymbolID USR = SymbolID();

  // Currently only used for namespaces and records.
  SymbolID ParentUSR = SymbolID();

  // InfoType of this particular Info.
  InfoType IT = InfoType::IT_default;

  // Comment description of this decl.
  OwningVec<CommentInfo> Description;

  SmallVector<Context, 4> Contexts;
};

inline Context::Context(const Info &I)
    : Reference(I.USR, I.Name, I.IT, I.Name, I.Path, I.DocumentationFileName) {}

// Info for namespaces.
struct NamespaceInfo : public Info {
  NamespaceInfo(SymbolID USR = SymbolID(), StringRef Name = StringRef(),
                StringRef Path = StringRef());

  void merge(NamespaceInfo &&I);

  ScopeChildren Children;
};

// Info for symbols.
struct SymbolInfo : public Info {
  SymbolInfo(InfoType IT, SymbolID USR = SymbolID(),
             StringRef Name = StringRef(), StringRef Path = StringRef())
      : Info(IT, USR, Name, Path) {}

  void merge(SymbolInfo &&I);

  bool operator<(const SymbolInfo &Other) const {
    // Sort by declaration location since we want the doc to be
    // generated in the order of the source code.
    // If the declaration location is the same, or not present
    // we sort by defined location otherwise fallback to the extracted name
    if (Loc.size() > 0 && Other.Loc.size() > 0 && Loc[0] != Other.Loc[0])
      return Loc[0] < Other.Loc[0];

    if (DefLoc && Other.DefLoc && *DefLoc != *Other.DefLoc)
      return *DefLoc < *Other.DefLoc;

    return extractName() < Other.extractName();
  }

  std::optional<Location> DefLoc;     // Location where this decl is defined.
  llvm::SmallVector<Location, 2> Loc; // Locations where this decl is declared.
  StringRef MangledName;
  bool IsStatic = false;
};

struct FriendInfo : public SymbolInfo, public llvm::ilist_node<FriendInfo> {
  FriendInfo() : SymbolInfo(InfoType::IT_friend) {}
  FriendInfo(SymbolID USR) : SymbolInfo(InfoType::IT_friend, USR) {}
  FriendInfo(const InfoType IT, const SymbolID &USR,
             const StringRef Name = StringRef())
      : SymbolInfo(IT, USR, Name) {}
  bool mergeable(const FriendInfo &Other);
  void merge(FriendInfo &&Other);

  Reference Ref;
  std::optional<TemplateInfo> Template;
  std::optional<TypeInfo> ReturnType;
  llvm::ArrayRef<FieldTypeInfo> Params;
  bool IsClass = false;
};

struct VarInfo : public SymbolInfo, public llvm::ilist_node<VarInfo> {
  VarInfo() : SymbolInfo(InfoType::IT_variable) {}
  explicit VarInfo(SymbolID USR) : SymbolInfo(InfoType::IT_variable, USR) {}

  void merge(VarInfo &&I);

  TypeInfo Type;
};

// TODO: Expand to allow for documenting templating and default args.
// Info for functions.
struct FunctionInfo : public SymbolInfo, public llvm::ilist_node<FunctionInfo> {
  FunctionInfo(SymbolID USR = SymbolID())
      : SymbolInfo(InfoType::IT_function, USR) {}

  void merge(FunctionInfo &&I);

  Reference Parent;
  TypeInfo ReturnType;
  llvm::SmallVector<FieldTypeInfo, 4> Params;
  StringRef Prototype;

  // When present, this function is a template or specialization.
  std::optional<TemplateInfo> Template;

  // Access level for this method (public, private, protected, none).
  // AS_public is set as default because the bitcode writer requires the enum
  // with value 0 to be used as the default.
  // (AS_public = 0, AS_protected = 1, AS_private = 2, AS_none = 3)
  AccessSpecifier Access = AccessSpecifier::AS_public;

  bool IsMethod = false;
};

// TODO: Expand to allow for documenting templating, inheritance access,
// friend classes
// Info for types.
struct RecordInfo : public SymbolInfo {
  RecordInfo(SymbolID USR = SymbolID(), StringRef Name = StringRef(),
             StringRef Path = StringRef());

  void merge(RecordInfo &&I);

  // Type of this record (struct, class, union, interface).
  TagTypeKind TagType = TagTypeKind::Struct;

  // Indicates if the record was declared using a typedef. Things like anonymous
  // structs in a typedef:
  //   typedef struct { ... } foo_t;
  // are converted into records with the typedef as the Name + this flag set.
  bool IsTypeDef = false;

  // When present, this record is a template or specialization.
  std::optional<TemplateInfo> Template;

  llvm::SmallVector<MemberTypeInfo, 4>
      Members;                             // List of info about record members.
  llvm::SmallVector<Reference, 4> Parents; // List of base/parent records
                                           // (does not include virtual
                                           // parents).
  llvm::SmallVector<Reference, 4>
      VirtualParents; // List of virtual base/parent records.

  OwningVec<BaseRecordInfo> Bases; // List of base/parent records; this includes
                                   // inherited methods and attributes

  OwningVec<FriendInfo> Friends;

  ScopeChildren Children;
};

// Info for typedef and using statements.
struct TypedefInfo : public SymbolInfo, public llvm::ilist_node<TypedefInfo> {
  TypedefInfo(SymbolID USR = SymbolID())
      : SymbolInfo(InfoType::IT_typedef, USR) {}

  void merge(TypedefInfo &&I);

  TypeInfo Underlying;

  // Only type aliases can be templates.
  std::optional<TemplateInfo> Template;

  // Underlying type declaration
  StringRef TypeDeclaration;

  // Indicates if this is a new C++ "using"-style typedef:
  //   using MyVector = std::vector<int>
  // False means it's a C-style typedef:
  //   typedef std::vector<int> MyVector;
  bool IsUsing = false;
};

struct BaseRecordInfo : public RecordInfo {
  BaseRecordInfo();
  BaseRecordInfo(SymbolID USR, StringRef Name, StringRef Path, bool IsVirtual,
                 AccessSpecifier Access, bool IsParent);

  // Access level associated with this inherited info (public, protected,
  // private).
  AccessSpecifier Access = AccessSpecifier::AS_public;
  // Indicates if base corresponds to a virtual inheritance
  bool IsVirtual = false;
  bool IsParent = false; // Indicates if this base is a direct parent
};

// Information for a single possible value of an enumeration.
struct EnumValueInfo {
  explicit EnumValueInfo(StringRef Name = StringRef(),
                         StringRef Value = StringRef("0"),
                         StringRef ValueExpr = StringRef())
      : Name(internString(Name)), Value(internString(Value)),
        ValueExpr(internString(ValueExpr)) {}

  bool operator==(const EnumValueInfo &Other) const {
    return std::tie(Name, Value, ValueExpr) ==
           std::tie(Other.Name, Other.Value, Other.ValueExpr);
  }

  StringRef Name;

  // The computed value of the enumeration constant. This could be the result of
  // evaluating the ValueExpr, or it could be automatically generated according
  // to C rules.
  StringRef Value;

  // Stores the user-supplied initialization expression for this enumeration
  // constant. This will be empty for implicit enumeration values.
  StringRef ValueExpr;

  /// Comment description of this field.
  OwningVec<CommentInfo> Description;
};

// TODO: Expand to allow for documenting templating.
// Info for types.
struct EnumInfo : public SymbolInfo, public llvm::ilist_node<EnumInfo> {
  EnumInfo() : SymbolInfo(InfoType::IT_enum) {}
  EnumInfo(SymbolID USR) : SymbolInfo(InfoType::IT_enum, USR) {}

  void merge(EnumInfo &&I);

  // Indicates whether this enum is scoped (e.g. enum class).
  bool Scoped = false;

  // Set to nonempty to the type when this is an explicitly typed enum. For
  //   enum Foo : short { ... };
  // this will be "short".
  std::optional<TypeInfo> BaseType;

  llvm::SmallVector<EnumValueInfo, 4> Members; // List of enum members.
};

struct ConceptInfo : public SymbolInfo, public llvm::ilist_node<ConceptInfo> {
  ConceptInfo() : SymbolInfo(InfoType::IT_concept) {}
  ConceptInfo(SymbolID USR) : SymbolInfo(InfoType::IT_concept, USR) {}

  void merge(ConceptInfo &&I);

  bool IsType;
  TemplateInfo Template;
  StringRef ConstraintExpression;
};

struct Index : public Reference {
  Index() = default;
  Index(StringRef Name) : Reference(SymbolID(), Name) {}
  Index(StringRef Name, StringRef JumpToSection)
      : Reference(SymbolID(), Name), JumpToSection(JumpToSection) {}
  Index(SymbolID USR, StringRef Name, InfoType IT, StringRef Path)
      : Reference(USR, Name, IT, Name, Path) {}
  // This is used to look for a USR in a vector of Indexes using std::find
  bool operator==(const SymbolID &Other) const { return USR == Other; }
  bool operator<(const Index &Other) const;

  std::optional<StringRef> JumpToSection;
  llvm::StringMap<Index> Children;

  OwningVec<const Index *> getSortedChildren() const;
  void sort();
};

// TODO: Add functionality to include separate markdown pages.

// A standalone function to call to merge a vector of infos into one.
// This assumes that all infos in the vector are of the same type, and will fail
// if they are different.
llvm::Expected<OwnedPtr<Info>> mergeInfos(OwningPtrArray<Info> &Values);

struct ClangDocContext {
  ClangDocContext(tooling::ExecutionContext *ECtx, StringRef ProjectName,
                  bool PublicOnly, StringRef OutDirectory, StringRef SourceRoot,
                  StringRef RepositoryUrl, StringRef RepositoryCodeLinePrefix,
                  StringRef Base, std::vector<std::string> UserStylesheets,
                  clang::DiagnosticsEngine &Diags, OutputFormatTy Format,
                  bool FTimeTrace = false);
  tooling::ExecutionContext *ECtx;
  std::string ProjectName;  // Name of project clang-doc is documenting.
  std::string OutDirectory; // Directory for outputting generated files.
  std::string SourceRoot;   // Directory where processed files are stored. Links
                            // to definition locations will only be generated if
                            // the file is in this dir.
  // URL of repository that hosts code used for links to definition locations.
  std::optional<std::string> RepositoryUrl;
  // Prefix of line code for repository.
  std::optional<std::string> RepositoryLinePrefix;
  // Path of CSS stylesheets that will be copied to OutDirectory and used to
  // style all HTML files.
  std::vector<std::string> UserStylesheets;
  // JavaScript files that will be imported in all HTML files.
  std::vector<std::string> JsScripts;
  // Base directory for remote repositories.
  StringRef Base;
  // Maps mustache template types to specific mustache template files.
  // Ex.    comment-template -> /path/to/comment-template.mustache
  llvm::StringMap<std::string> MustacheTemplates;
  // A pointer to a DiagnosticsEngine for error reporting.
  clang::DiagnosticsEngine &Diags;
  Index Idx;
  OutputFormatTy Format;
  int Granularity; // Granularity of ftime trace
  bool PublicOnly; // Indicates if only public declarations are documented.
  bool FTimeTrace; // Indicates if ftime trace is turned on
};

// Ensure arena allocated types remain safe to allocate in the arena.
// Only trivially destructible types are safe, so enforce that at compile-time.
static_assert(std::is_trivially_destructible_v<ConstraintInfo>);
static_assert(std::is_trivially_destructible_v<FieldTypeInfo>);
static_assert(std::is_trivially_destructible_v<Location>);
static_assert(std::is_trivially_destructible_v<Reference>);
static_assert(std::is_trivially_destructible_v<TemplateParamInfo>);
static_assert(std::is_trivially_destructible_v<TypeInfo>);

// FIXME: These types need to be trivially destructible for arena allocation.
static_assert(!std::is_trivially_destructible_v<CommentInfo>);
static_assert(!std::is_trivially_destructible_v<ConceptInfo>);
static_assert(!std::is_trivially_destructible_v<EnumInfo>);
static_assert(!std::is_trivially_destructible_v<FriendInfo>);
static_assert(!std::is_trivially_destructible_v<FunctionInfo>);
static_assert(!std::is_trivially_destructible_v<Info>);
static_assert(!std::is_trivially_destructible_v<MemberTypeInfo>);
static_assert(!std::is_trivially_destructible_v<NamespaceInfo>);
static_assert(!std::is_trivially_destructible_v<RecordInfo>);
static_assert(!std::is_trivially_destructible_v<ScopeChildren>);
static_assert(!std::is_trivially_destructible_v<SymbolInfo>);
static_assert(!std::is_trivially_destructible_v<TemplateInfo>);
static_assert(!std::is_trivially_destructible_v<TemplateSpecializationInfo>);
static_assert(!std::is_trivially_destructible_v<TypedefInfo>);
static_assert(!std::is_trivially_destructible_v<VarInfo>);

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_REPRESENTATION_H
