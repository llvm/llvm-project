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
#include "clang/Basic/Specifiers.h"
#include "clang/Tooling/StandaloneExecution.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <array>
#include <string>

namespace clang {
namespace doc {

// SHA1'd hash of a USR.
using SymbolID = std::array<uint8_t, 20>;

struct Info;
struct FunctionInfo;
struct EnumInfo;
struct BaseRecordInfo;

enum class InfoType {
  IT_default,
  IT_namespace,
  IT_record,
  IT_function,
  IT_enum
};

// A representation of a parsed comment.
struct CommentInfo {
  CommentInfo() = default;
  CommentInfo(CommentInfo &Other) = delete;
  CommentInfo(CommentInfo &&Other) = default;
  CommentInfo &operator=(CommentInfo &&Other) = default;

  bool operator==(const CommentInfo &Other) const {
    auto FirstCI = std::tie(Kind, Text, Name, Direction, ParamName, CloseName,
                            SelfClosing, Explicit, AttrKeys, AttrValues, Args);
    auto SecondCI =
        std::tie(Other.Kind, Other.Text, Other.Name, Other.Direction,
                 Other.ParamName, Other.CloseName, Other.SelfClosing,
                 Other.Explicit, Other.AttrKeys, Other.AttrValues, Other.Args);

    if (FirstCI != SecondCI || Children.size() != Other.Children.size())
      return false;

    return std::equal(Children.begin(), Children.end(), Other.Children.begin(),
                      llvm::deref<std::equal_to<>>{});
  }

  // This operator is used to sort a vector of CommentInfos.
  // No specific order (attributes more important than others) is required. Any
  // sort is enough, the order is only needed to call std::unique after sorting
  // the vector.
  bool operator<(const CommentInfo &Other) const {
    auto FirstCI = std::tie(Kind, Text, Name, Direction, ParamName, CloseName,
                            SelfClosing, Explicit, AttrKeys, AttrValues, Args);
    auto SecondCI =
        std::tie(Other.Kind, Other.Text, Other.Name, Other.Direction,
                 Other.ParamName, Other.CloseName, Other.SelfClosing,
                 Other.Explicit, Other.AttrKeys, Other.AttrValues, Other.Args);

    if (FirstCI < SecondCI)
      return true;

    if (FirstCI == SecondCI) {
      return std::lexicographical_compare(
          Children.begin(), Children.end(), Other.Children.begin(),
          Other.Children.end(), llvm::deref<std::less<>>());
    }

    return false;
  }

  SmallString<16>
      Kind; // Kind of comment (FullComment, ParagraphComment, TextComment,
            // InlineCommandComment, HTMLStartTagComment, HTMLEndTagComment,
            // BlockCommandComment, ParamCommandComment,
            // TParamCommandComment, VerbatimBlockComment,
            // VerbatimBlockLineComment, VerbatimLineComment).
  SmallString<64> Text;      // Text of the comment.
  SmallString<16> Name;      // Name of the comment (for Verbatim and HTML).
  SmallString<8> Direction;  // Parameter direction (for (T)ParamCommand).
  SmallString<16> ParamName; // Parameter name (for (T)ParamCommand).
  SmallString<16> CloseName; // Closing tag name (for VerbatimBlock).
  bool SelfClosing = false;  // Indicates if tag is self-closing (for HTML).
  bool Explicit = false; // Indicates if the direction of a param is explicit
                         // (for (T)ParamCommand).
  llvm::SmallVector<SmallString<16>, 4>
      AttrKeys; // List of attribute keys (for HTML).
  llvm::SmallVector<SmallString<16>, 4>
      AttrValues; // List of attribute values for each key (for HTML).
  llvm::SmallVector<SmallString<16>, 4>
      Args; // List of arguments to commands (for InlineCommand).
  std::vector<std::unique_ptr<CommentInfo>>
      Children; // List of child comments for this CommentInfo.
};

struct Reference {
  Reference() = default;
  Reference(llvm::StringRef Name) : Name(Name) {}
  // An empty path means the info is in the global namespace because the path is
  // a composite of the parent namespaces.
  Reference(llvm::StringRef Name, StringRef Path)
      : Name(Name), Path(Path), IsInGlobalNamespace(Path.empty()) {}
  Reference(SymbolID USR, StringRef Name, InfoType IT)
      : USR(USR), Name(Name), RefType(IT) {}
  // An empty path means the info is in the global namespace because the path is
  // a composite of the parent namespaces.
  Reference(SymbolID USR, StringRef Name, InfoType IT, StringRef Path)
      : USR(USR), Name(Name), RefType(IT), Path(Path),
        IsInGlobalNamespace(Path.empty()) {}

  bool operator==(const Reference &Other) const {
    return std::tie(USR, Name, RefType) ==
           std::tie(Other.USR, Other.Name, Other.RefType);
  }

  bool mergeable(const Reference &Other);
  void merge(Reference &&I);

  /// Returns the path for this Reference relative to CurrentPath.
  llvm::SmallString<64> getRelativeFilePath(const StringRef &CurrentPath) const;

  /// Returns the basename that should be used for this Reference.
  llvm::SmallString<16> getFileBaseName() const;

  SymbolID USR = SymbolID(); // Unique identifier for referenced decl
  SmallString<16> Name;      // Name of type (possibly unresolved).
  InfoType RefType = InfoType::IT_default; // Indicates the type of this
                                           // Reference (namespace, record,
                                           // function, enum, default).
  // Path of directory where the clang-doc generated file will be saved
  // (possibly unresolved)
  llvm::SmallString<128> Path;
  // Indicates if the info's parent is the global namespace, or if the info is
  // the global namespace
  bool IsInGlobalNamespace = false;
};

// A base struct for TypeInfos
struct TypeInfo {
  TypeInfo() = default;
  TypeInfo(SymbolID Type, StringRef Field, InfoType IT)
      : Type(Type, Field, IT) {}
  TypeInfo(SymbolID Type, StringRef Field, InfoType IT, StringRef Path)
      : Type(Type, Field, IT, Path) {}
  TypeInfo(llvm::StringRef RefName) : Type(RefName) {}
  TypeInfo(llvm::StringRef RefName, StringRef Path) : Type(RefName, Path) {}

  bool operator==(const TypeInfo &Other) const { return Type == Other.Type; }

  Reference Type; // Referenced type in this info.
};

// Info for field types.
struct FieldTypeInfo : public TypeInfo {
  FieldTypeInfo() = default;
  FieldTypeInfo(SymbolID Type, StringRef Field, InfoType IT, StringRef Path,
                llvm::StringRef Name)
      : TypeInfo(Type, Field, IT, Path), Name(Name) {}
  FieldTypeInfo(llvm::StringRef RefName, llvm::StringRef Name)
      : TypeInfo(RefName), Name(Name) {}
  FieldTypeInfo(llvm::StringRef RefName, StringRef Path, llvm::StringRef Name)
      : TypeInfo(RefName, Path), Name(Name) {}

  bool operator==(const FieldTypeInfo &Other) const {
    return std::tie(Type, Name) == std::tie(Other.Type, Other.Name);
  }

  SmallString<16> Name; // Name associated with this info.
};

// Info for member types.
struct MemberTypeInfo : public FieldTypeInfo {
  MemberTypeInfo() = default;
  MemberTypeInfo(SymbolID Type, StringRef Field, InfoType IT, StringRef Path,
                 llvm::StringRef Name, AccessSpecifier Access)
      : FieldTypeInfo(Type, Field, IT, Path, Name), Access(Access) {}
  MemberTypeInfo(llvm::StringRef RefName, llvm::StringRef Name,
                 AccessSpecifier Access)
      : FieldTypeInfo(RefName, Name), Access(Access) {}
  MemberTypeInfo(llvm::StringRef RefName, StringRef Path, llvm::StringRef Name,
                 AccessSpecifier Access)
      : FieldTypeInfo(RefName, Path, Name), Access(Access) {}

  bool operator==(const MemberTypeInfo &Other) const {
    return std::tie(Type, Name, Access) ==
           std::tie(Other.Type, Other.Name, Other.Access);
  }

  // Access level associated with this info (public, protected, private, none).
  // AS_public is set as default because the bitcode writer requires the enum
  // with value 0 to be used as the default.
  // (AS_public = 0, AS_protected = 1, AS_private = 2, AS_none = 3)
  AccessSpecifier Access = AccessSpecifier::AS_public;
};

struct Location {
  Location() = default;
  Location(int LineNumber, SmallString<16> Filename)
      : LineNumber(LineNumber), Filename(std::move(Filename)) {}
  Location(int LineNumber, SmallString<16> Filename, bool IsFileInRootDir)
      : LineNumber(LineNumber), Filename(std::move(Filename)),
        IsFileInRootDir(IsFileInRootDir) {}

  bool operator==(const Location &Other) const {
    return std::tie(LineNumber, Filename) ==
           std::tie(Other.LineNumber, Other.Filename);
  }

  // This operator is used to sort a vector of Locations.
  // No specific order (attributes more important than others) is required. Any
  // sort is enough, the order is only needed to call std::unique after sorting
  // the vector.
  bool operator<(const Location &Other) const {
    return std::tie(LineNumber, Filename) <
           std::tie(Other.LineNumber, Other.Filename);
  }

  int LineNumber;               // Line number of this Location.
  SmallString<32> Filename;     // File for this Location.
  bool IsFileInRootDir = false; // Indicates if file is inside root directory
};

/// A base struct for Infos.
struct Info {
  Info() = default;
  Info(InfoType IT) : IT(IT) {}
  Info(InfoType IT, SymbolID USR) : USR(USR), IT(IT) {}
  Info(InfoType IT, SymbolID USR, StringRef Name)
      : USR(USR), IT(IT), Name(Name) {}
  Info(InfoType IT, SymbolID USR, StringRef Name, StringRef Path)
      : USR(USR), IT(IT), Name(Name), Path(Path) {}
  Info(const Info &Other) = delete;
  Info(Info &&Other) = default;

  virtual ~Info() = default;

  SymbolID USR =
      SymbolID(); // Unique identifier for the decl described by this Info.
  const InfoType IT = InfoType::IT_default; // InfoType of this particular Info.
  SmallString<16> Name;                     // Unqualified name of the decl.
  llvm::SmallVector<Reference, 4>
      Namespace; // List of parent namespaces for this decl.
  std::vector<CommentInfo> Description; // Comment description of this decl.
  llvm::SmallString<128> Path;          // Path of directory where the clang-doc
                                        // generated file will be saved

  void mergeBase(Info &&I);
  bool mergeable(const Info &Other);

  llvm::SmallString<16> extractName() const;

  /// Returns the file path for this Info relative to CurrentPath.
  llvm::SmallString<64> getRelativeFilePath(const StringRef &CurrentPath) const;

  /// Returns the basename that should be used for this Info.
  llvm::SmallString<16> getFileBaseName() const;

  // Returns a reference to the parent scope (that is, the immediate parent
  // namespace or class in which this decl resides).
  llvm::Expected<Reference> getEnclosingScope();
};

// Info for namespaces.
struct NamespaceInfo : public Info {
  NamespaceInfo() : Info(InfoType::IT_namespace) {}
  NamespaceInfo(SymbolID USR) : Info(InfoType::IT_namespace, USR) {}
  NamespaceInfo(SymbolID USR, StringRef Name)
      : Info(InfoType::IT_namespace, USR, Name) {}
  NamespaceInfo(SymbolID USR, StringRef Name, StringRef Path)
      : Info(InfoType::IT_namespace, USR, Name, Path) {}

  void merge(NamespaceInfo &&I);

  // Namespaces and Records are references because they will be properly
  // documented in their own info, while the entirety of Functions and Enums are
  // included here because they should not have separate documentation from
  // their scope.
  std::vector<Reference> ChildNamespaces;
  std::vector<Reference> ChildRecords;
  std::vector<FunctionInfo> ChildFunctions;
  std::vector<EnumInfo> ChildEnums;
};

// Info for symbols.
struct SymbolInfo : public Info {
  SymbolInfo(InfoType IT) : Info(IT) {}
  SymbolInfo(InfoType IT, SymbolID USR) : Info(IT, USR) {}
  SymbolInfo(InfoType IT, SymbolID USR, StringRef Name) : Info(IT, USR, Name) {}
  SymbolInfo(InfoType IT, SymbolID USR, StringRef Name, StringRef Path)
      : Info(IT, USR, Name, Path) {}

  void merge(SymbolInfo &&I);

  llvm::Optional<Location> DefLoc;    // Location where this decl is defined.
  llvm::SmallVector<Location, 2> Loc; // Locations where this decl is declared.
};

// TODO: Expand to allow for documenting templating and default args.
// Info for functions.
struct FunctionInfo : public SymbolInfo {
  FunctionInfo() : SymbolInfo(InfoType::IT_function) {}
  FunctionInfo(SymbolID USR) : SymbolInfo(InfoType::IT_function, USR) {}

  void merge(FunctionInfo &&I);

  bool IsMethod = false; // Indicates whether this function is a class method.
  Reference Parent;      // Reference to the parent class decl for this method.
  TypeInfo ReturnType;   // Info about the return type of this function.
  llvm::SmallVector<FieldTypeInfo, 4> Params; // List of parameters.
  // Access level for this method (public, private, protected, none).
  // AS_public is set as default because the bitcode writer requires the enum
  // with value 0 to be used as the default.
  // (AS_public = 0, AS_protected = 1, AS_private = 2, AS_none = 3)
  AccessSpecifier Access = AccessSpecifier::AS_public;
};

// TODO: Expand to allow for documenting templating, inheritance access,
// friend classes
// Info for types.
struct RecordInfo : public SymbolInfo {
  RecordInfo() : SymbolInfo(InfoType::IT_record) {}
  RecordInfo(SymbolID USR) : SymbolInfo(InfoType::IT_record, USR) {}
  RecordInfo(SymbolID USR, StringRef Name)
      : SymbolInfo(InfoType::IT_record, USR, Name) {}
  RecordInfo(SymbolID USR, StringRef Name, StringRef Path)
      : SymbolInfo(InfoType::IT_record, USR, Name, Path) {}

  void merge(RecordInfo &&I);

  TagTypeKind TagType = TagTypeKind::TTK_Struct; // Type of this record
                                                 // (struct, class, union,
                                                 // interface).
  bool IsTypeDef = false; // Indicates if record was declared using typedef
  llvm::SmallVector<MemberTypeInfo, 4>
      Members;                             // List of info about record members.
  llvm::SmallVector<Reference, 4> Parents; // List of base/parent records
                                           // (does not include virtual
                                           // parents).
  llvm::SmallVector<Reference, 4>
      VirtualParents; // List of virtual base/parent records.

  std::vector<BaseRecordInfo>
      Bases; // List of base/parent records; this includes inherited methods and
             // attributes

  // Records are references because they will be properly documented in their
  // own info, while the entirety of Functions and Enums are included here
  // because they should not have separate documentation from their scope.
  std::vector<Reference> ChildRecords;
  std::vector<FunctionInfo> ChildFunctions;
  std::vector<EnumInfo> ChildEnums;
};

struct BaseRecordInfo : public RecordInfo {
  BaseRecordInfo() : RecordInfo() {}
  BaseRecordInfo(SymbolID USR, StringRef Name, StringRef Path, bool IsVirtual,
                 AccessSpecifier Access, bool IsParent)
      : RecordInfo(USR, Name, Path), IsVirtual(IsVirtual), Access(Access),
        IsParent(IsParent) {}

  // Indicates if base corresponds to a virtual inheritance
  bool IsVirtual = false;
  // Access level associated with this inherited info (public, protected,
  // private).
  AccessSpecifier Access = AccessSpecifier::AS_public;
  bool IsParent = false; // Indicates if this base is a direct parent
};

// TODO: Expand to allow for documenting templating.
// Info for types.
struct EnumInfo : public SymbolInfo {
  EnumInfo() : SymbolInfo(InfoType::IT_enum) {}
  EnumInfo(SymbolID USR) : SymbolInfo(InfoType::IT_enum, USR) {}

  void merge(EnumInfo &&I);

  bool Scoped =
      false; // Indicates whether this enum is scoped (e.g. enum class).
  llvm::SmallVector<SmallString<16>, 4> Members; // List of enum members.
};

struct Index : public Reference {
  Index() = default;
  Index(StringRef Name) : Reference(Name) {}
  Index(StringRef Name, StringRef JumpToSection)
      : Reference(Name), JumpToSection(JumpToSection) {}
  Index(SymbolID USR, StringRef Name, InfoType IT, StringRef Path)
      : Reference(USR, Name, IT, Path) {}
  // This is used to look for a USR in a vector of Indexes using std::find
  bool operator==(const SymbolID &Other) const { return USR == Other; }
  bool operator<(const Index &Other) const;

  llvm::Optional<SmallString<16>> JumpToSection;
  std::vector<Index> Children;

  void sort();
};

// TODO: Add functionality to include separate markdown pages.

// A standalone function to call to merge a vector of infos into one.
// This assumes that all infos in the vector are of the same type, and will fail
// if they are different.
llvm::Expected<std::unique_ptr<Info>>
mergeInfos(std::vector<std::unique_ptr<Info>> &Values);

struct ClangDocContext {
  ClangDocContext() = default;
  ClangDocContext(tooling::ExecutionContext *ECtx, StringRef ProjectName,
                  bool PublicOnly, StringRef OutDirectory, StringRef SourceRoot,
                  StringRef RepositoryUrl,
                  std::vector<std::string> UserStylesheets,
                  std::vector<std::string> JsScripts);
  tooling::ExecutionContext *ECtx;
  std::string ProjectName; // Name of project clang-doc is documenting.
  bool PublicOnly; // Indicates if only public declarations are documented.
  std::string OutDirectory; // Directory for outputting generated files.
  std::string SourceRoot;   // Directory where processed files are stored. Links
                            // to definition locations will only be generated if
                            // the file is in this dir.
  // URL of repository that hosts code used for links to definition locations.
  llvm::Optional<std::string> RepositoryUrl;
  // Path of CSS stylesheets that will be copied to OutDirectory and used to
  // style all HTML files.
  std::vector<std::string> UserStylesheets;
  // JavaScript files that will be imported in allHTML file.
  std::vector<std::string> JsScripts;
  // Other files that should be copied to OutDirectory, besides UserStylesheets.
  std::vector<std::string> FilesToCopy;
  Index Idx;
};

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_REPRESENTATION_H
