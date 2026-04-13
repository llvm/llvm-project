//===-- YAMLGenerator.cpp - ClangDoc YAML -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of the YAML generator, converting decl info into YAML output.
//===----------------------------------------------------------------------===//

#include "Generators.h"
#include "Representation.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace clang::doc;

// These define YAML traits for decoding the listed values within a vector.
LLVM_YAML_IS_SEQUENCE_VECTOR(FieldTypeInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(MemberTypeInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(Reference)
LLVM_YAML_IS_SEQUENCE_VECTOR(Location)
LLVM_YAML_IS_SEQUENCE_VECTOR(CommentInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(FunctionInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(EnumInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(EnumValueInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(TemplateParamInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(TypedefInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(BaseRecordInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(OwnedPtr<CommentInfo>)

namespace llvm {

template <typename T>
bool operator==(const llvm::simple_ilist<T> &LHS,
                const llvm::simple_ilist<T> &RHS) {
  auto LIt = LHS.begin(), LEnd = LHS.end();
  auto RIt = RHS.begin(), REnd = RHS.end();
  for (; LIt != LEnd && RIt != REnd; ++LIt, ++RIt) {
    if (!(*LIt == *RIt))
      return false;
  }
  return LIt == LEnd && RIt == REnd;
}

template <typename T>
bool operator!=(const llvm::simple_ilist<T> &LHS,
                const llvm::simple_ilist<T> &RHS) {
  return !(LHS == RHS);
}

namespace yaml {

// Provide SequenceTraits for ArrayRef<T*> since YAMLTraits only provides it for
// MutableArrayRef
template <typename T> struct SequenceTraits<ArrayRef<T *>> {
  static size_t size(IO &io, ArrayRef<T *> &seq) { return seq.size(); }
  static T *&element(IO &io, ArrayRef<T *> &seq, size_t index) {
    // ArrayRef is not mutable, but YAML output only reads the value.
    return const_cast<T *&>(seq[index]);
  }
};

template <typename T> struct SequenceTraits<llvm::simple_ilist<T>> {
  static size_t size(IO &io, llvm::simple_ilist<T> &seq) { return seq.size(); }
  static T &element(IO &io, llvm::simple_ilist<T> &seq, size_t index) {
    return *std::next(seq.begin(), index);
  }
};

// Map pointers to the value mappings as clang-doc only does output
// serialization.
template <typename T> struct PointerMappingTraits {
  static void mapping(IO &IO, T *&Val) {
    if (Val)
      MappingTraits<T>::mapping(IO, *Val);
  }
};

template <>
struct MappingTraits<clang::doc::Reference *>
    : PointerMappingTraits<clang::doc::Reference> {};
template <>
struct MappingTraits<clang::doc::CommentInfo *>
    : PointerMappingTraits<clang::doc::CommentInfo> {};
template <>
struct MappingTraits<clang::doc::FunctionInfo *>
    : PointerMappingTraits<clang::doc::FunctionInfo> {};
template <>
struct MappingTraits<clang::doc::EnumInfo *>
    : PointerMappingTraits<clang::doc::EnumInfo> {};
template <>
struct MappingTraits<clang::doc::TemplateParamInfo *>
    : PointerMappingTraits<clang::doc::TemplateParamInfo> {};

template <typename T> struct SequenceTraits<ArrayRef<T>> {
  static size_t size(IO &io, llvm::ArrayRef<T> &seq) { return seq.size(); }
  static T &element(IO &io, llvm::ArrayRef<T> &seq, size_t index) {
    return const_cast<T &>(seq[index]);
  }
};

// Enumerations to YAML output.

template <> struct ScalarEnumerationTraits<clang::AccessSpecifier> {
  static void enumeration(IO &IO, clang::AccessSpecifier &Value) {
    IO.enumCase(Value, "Public", clang::AccessSpecifier::AS_public);
    IO.enumCase(Value, "Protected", clang::AccessSpecifier::AS_protected);
    IO.enumCase(Value, "Private", clang::AccessSpecifier::AS_private);
    IO.enumCase(Value, "None", clang::AccessSpecifier::AS_none);
  }
};

template <> struct ScalarEnumerationTraits<clang::TagTypeKind> {
  static void enumeration(IO &IO, clang::TagTypeKind &Value) {
    IO.enumCase(Value, "Struct", clang::TagTypeKind::Struct);
    IO.enumCase(Value, "Interface", clang::TagTypeKind::Interface);
    IO.enumCase(Value, "Union", clang::TagTypeKind::Union);
    IO.enumCase(Value, "Class", clang::TagTypeKind::Class);
    IO.enumCase(Value, "Enum", clang::TagTypeKind::Enum);
  }
};

template <> struct ScalarEnumerationTraits<InfoType> {
  static void enumeration(IO &IO, InfoType &Value) {
    IO.enumCase(Value, "Namespace", InfoType::IT_namespace);
    IO.enumCase(Value, "Record", InfoType::IT_record);
    IO.enumCase(Value, "Function", InfoType::IT_function);
    IO.enumCase(Value, "Enum", InfoType::IT_enum);
    IO.enumCase(Value, "Default", InfoType::IT_default);
  }
};

template <> struct ScalarEnumerationTraits<clang::doc::CommentKind> {
  static void enumeration(IO &IO, clang::doc::CommentKind &Value) {
    IO.enumCase(Value, "FullComment", clang::doc::CommentKind::CK_FullComment);
    IO.enumCase(Value, "ParagraphComment",
                clang::doc::CommentKind::CK_ParagraphComment);
    IO.enumCase(Value, "TextComment", clang::doc::CommentKind::CK_TextComment);
    IO.enumCase(Value, "InlineCommandComment",
                clang::doc::CommentKind::CK_InlineCommandComment);
    IO.enumCase(Value, "HTMLStartTagComment",
                clang::doc::CommentKind::CK_HTMLStartTagComment);
    IO.enumCase(Value, "HTMLEndTagComment",
                clang::doc::CommentKind::CK_HTMLEndTagComment);
    IO.enumCase(Value, "BlockCommandComment",
                clang::doc::CommentKind::CK_BlockCommandComment);
    IO.enumCase(Value, "ParamCommandComment",
                clang::doc::CommentKind::CK_ParamCommandComment);
    IO.enumCase(Value, "TParamCommandComment",
                clang::doc::CommentKind::CK_TParamCommandComment);
    IO.enumCase(Value, "VerbatimBlockComment",
                clang::doc::CommentKind::CK_VerbatimBlockComment);
    IO.enumCase(Value, "VerbatimBlockLineComment",
                clang::doc::CommentKind::CK_VerbatimBlockLineComment);
    IO.enumCase(Value, "VerbatimLineComment",
                clang::doc::CommentKind::CK_VerbatimLineComment);
    IO.enumCase(Value, "Unknown", clang::doc::CommentKind::CK_Unknown);
  }
};

// Scalars to YAML output.

template <> struct ScalarTraits<SymbolID> {

  static void output(const SymbolID &S, void *, llvm::raw_ostream &OS) {
    OS << toHex(toStringRef(S));
  }

  static StringRef input(StringRef Scalar, void *, SymbolID &Value) {
    if (Scalar.size() != 40)
      return "Error: Incorrect scalar size for USR.";
    Value = stringToSymbol(Scalar);
    return StringRef();
  }

  static SymbolID stringToSymbol(llvm::StringRef Value) {
    SymbolID USR;
    std::string HexString = fromHex(Value);
    std::copy(HexString.begin(), HexString.end(), USR.begin());
    return SymbolID(USR);
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Single; }
};

/// A wrapper for StringRef to force YAML traits to single-quote the string.
struct QuotedString {
  StringRef Ref;
  QuotedString() = default;
  QuotedString(StringRef R) : Ref(R) {}
  operator StringRef() const { return Ref; }
  bool operator==(const QuotedString &Other) const { return Ref == Other.Ref; }
};

template <> struct ScalarTraits<QuotedString> {
  static void output(const QuotedString &S, void *, llvm::raw_ostream &OS) {
    OS << S.Ref;
  }
  static StringRef input(StringRef Scalar, void *, QuotedString &Value) {
    Value.Ref = Scalar;
    return StringRef();
  }
  static QuotingType mustQuote(StringRef) { return QuotingType::Single; }
};
} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::yaml::QuotedString)

namespace llvm {
namespace yaml {

// Helper functions to map infos to YAML.

static void typeInfoMapping(IO &IO, TypeInfo &I) {
  IO.mapOptional("Type", I.Type, Reference());
}

static void fieldTypeInfoMapping(IO &IO, FieldTypeInfo &I) {
  typeInfoMapping(IO, I);

  QuotedString QName(I.Name);
  IO.mapOptional("Name", QName, QuotedString(StringRef()));
  if (!IO.outputting())
    I.Name = QName.Ref;

  QuotedString QDefault(I.DefaultValue);
  IO.mapOptional("DefaultValue", QDefault, QuotedString(StringRef()));
  if (!IO.outputting())
    I.DefaultValue = QDefault.Ref;
}

static void infoMapping(IO &IO, Info &I) {
  IO.mapRequired("USR", I.USR);

  QuotedString QName(I.Name);
  IO.mapOptional("Name", QName, QuotedString(StringRef()));
  if (!IO.outputting())
    I.Name = QName.Ref;

  QuotedString QPath(I.Path);
  IO.mapOptional("Path", QPath, QuotedString(StringRef()));
  if (!IO.outputting())
    I.Path = QPath.Ref;

  IO.mapOptional("Namespace", I.Namespace, llvm::SmallVector<Reference, 4>());
  IO.mapOptional("Description", I.Description);
}

static void symbolInfoMapping(IO &IO, SymbolInfo &I) {
  infoMapping(IO, I);
  IO.mapOptional("DefLocation", I.DefLoc, std::optional<Location>());
  IO.mapOptional("Location", I.Loc);
}

static void recordInfoMapping(IO &IO, RecordInfo &I) {
  symbolInfoMapping(IO, I);
  IO.mapOptional("TagType", I.TagType);
  IO.mapOptional("IsTypeDef", I.IsTypeDef, false);
  IO.mapOptional("Members", I.Members);
  IO.mapOptional("Bases", I.Bases);
  IO.mapOptional("Parents", I.Parents, SmallVector<Reference, 4>());
  IO.mapOptional("VirtualParents", I.VirtualParents,
                 llvm::SmallVector<Reference, 4>());
  IO.mapOptional("ChildRecords", I.Children.Records);
  IO.mapOptional("ChildFunctions", I.Children.Functions);
  IO.mapOptional("ChildEnums", I.Children.Enums);
  IO.mapOptional("ChildTypedefs", I.Children.Typedefs);
  IO.mapOptional("Template", I.Template);
}

static void commentInfoMapping(IO &IO, CommentInfo &I) {
  IO.mapOptional("Kind", I.Kind, CommentKind::CK_Unknown);

  QuotedString QText(I.Text);
  IO.mapOptional("Text", QText, QuotedString(StringRef()));
  if (!IO.outputting())
    I.Text = QText.Ref;

  QuotedString QName(I.Name);
  IO.mapOptional("Name", QName, QuotedString(StringRef()));
  if (!IO.outputting())
    I.Name = QName.Ref;

  QuotedString QDirection(I.Direction);
  IO.mapOptional("Direction", QDirection, QuotedString(StringRef()));
  if (!IO.outputting())
    I.Direction = QDirection.Ref;

  QuotedString QParamName(I.ParamName);
  IO.mapOptional("ParamName", QParamName, QuotedString(StringRef()));
  if (!IO.outputting())
    I.ParamName = QParamName.Ref;

  QuotedString QCloseName(I.CloseName);
  IO.mapOptional("CloseName", QCloseName, QuotedString(StringRef()));
  if (!IO.outputting())
    I.CloseName = QCloseName.Ref;

  IO.mapOptional("SelfClosing", I.SelfClosing, false);
  IO.mapOptional("Explicit", I.Explicit, false);

  std::vector<QuotedString> QArgs;
  if (IO.outputting()) {
    for (auto &S : I.Args)
      QArgs.push_back(QuotedString(S));
  }
  IO.mapOptional("Args", QArgs, std::vector<QuotedString>());

  std::vector<QuotedString> QAttrKeys;
  if (IO.outputting()) {
    for (auto &S : I.AttrKeys)
      QAttrKeys.push_back(QuotedString(S));
  }
  IO.mapOptional("AttrKeys", QAttrKeys, std::vector<QuotedString>());

  std::vector<QuotedString> QAttrValues;
  if (IO.outputting()) {
    for (auto &S : I.AttrValues)
      QAttrValues.push_back(QuotedString(S));
  }
  IO.mapOptional("AttrValues", QAttrValues, std::vector<QuotedString>());

  IO.mapOptional("Children", I.Children);
}

// Template specialization to YAML traits for Infos.

template <> struct MappingTraits<Location> {
  static void mapping(IO &IO, Location &Loc) {
    IO.mapOptional("LineNumber", Loc.StartLineNumber, 0);

    QuotedString QFilename(Loc.Filename);
    IO.mapOptional("Filename", QFilename, QuotedString(StringRef()));
    if (!IO.outputting())
      Loc.Filename = QFilename.Ref;
  }
};

template <> struct MappingTraits<Reference> {
  static void mapping(IO &IO, Reference &Ref) {
    IO.mapOptional("Type", Ref.RefType, InfoType::IT_default);

    QuotedString QName(Ref.Name);
    IO.mapOptional("Name", QName, QuotedString(StringRef()));
    if (!IO.outputting())
      Ref.Name = QName.Ref;

    QuotedString QQualName(Ref.QualName);
    IO.mapOptional("QualName", QQualName, QuotedString(StringRef()));
    if (!IO.outputting())
      Ref.QualName = QQualName.Ref;

    IO.mapOptional("USR", Ref.USR, SymbolID());

    QuotedString QPath(Ref.Path);
    IO.mapOptional("Path", QPath, QuotedString(StringRef()));
    if (!IO.outputting())
      Ref.Path = QPath.Ref;
  }
};

template <> struct MappingTraits<TypeInfo> {
  static void mapping(IO &IO, TypeInfo &I) { typeInfoMapping(IO, I); }
};

template <> struct MappingTraits<FieldTypeInfo> {
  static void mapping(IO &IO, FieldTypeInfo &I) {
    typeInfoMapping(IO, I);

    QuotedString QName(I.Name);
    IO.mapOptional("Name", QName, QuotedString(StringRef()));
    if (!IO.outputting())
      I.Name = QName.Ref;

    QuotedString QDefault(I.DefaultValue);
    IO.mapOptional("DefaultValue", QDefault, QuotedString(StringRef()));
    if (!IO.outputting())
      I.DefaultValue = QDefault.Ref;
  }
};

template <> struct MappingTraits<MemberTypeInfo> {
  static void mapping(IO &IO, MemberTypeInfo &I) {
    fieldTypeInfoMapping(IO, I);
    // clang::AccessSpecifier::AS_none is used as the default here because it's
    // the AS that shouldn't be part of the output. Even though AS_public is the
    // default in the struct, it should be displayed in the YAML output.
    IO.mapOptional("Access", I.Access, clang::AccessSpecifier::AS_none);
    IO.mapOptional("Description", I.Description);
  }
};

template <> struct MappingTraits<NamespaceInfo> {
  static void mapping(IO &IO, NamespaceInfo &I) {
    infoMapping(IO, I);
    std::vector<Reference> TempNamespaces;
    for (const auto &N : I.Children.Namespaces)
      TempNamespaces.push_back(N);
    IO.mapOptional("ChildNamespaces", TempNamespaces, std::vector<Reference>());
    IO.mapOptional("ChildRecords", I.Children.Records);
    IO.mapOptional("ChildFunctions", I.Children.Functions);
    IO.mapOptional("ChildEnums", I.Children.Enums);
    IO.mapOptional("ChildTypedefs", I.Children.Typedefs);
  }
};

template <> struct MappingTraits<RecordInfo> {
  static void mapping(IO &IO, RecordInfo &I) { recordInfoMapping(IO, I); }
};

template <> struct MappingTraits<BaseRecordInfo> {
  static void mapping(IO &IO, BaseRecordInfo &I) {
    recordInfoMapping(IO, I);
    IO.mapOptional("IsVirtual", I.IsVirtual, false);
    // clang::AccessSpecifier::AS_none is used as the default here because it's
    // the AS that shouldn't be part of the output. Even though AS_public is the
    // default in the struct, it should be displayed in the YAML output.
    IO.mapOptional("Access", I.Access, clang::AccessSpecifier::AS_none);
    IO.mapOptional("IsParent", I.IsParent, false);
  }
};

template <> struct MappingTraits<EnumValueInfo> {
  static void mapping(IO &IO, EnumValueInfo &I) {
    QuotedString QName(I.Name);
    IO.mapOptional("Name", QName, QuotedString(StringRef()));
    if (!IO.outputting())
      I.Name = QName.Ref;

    QuotedString QValue(I.Value);
    IO.mapOptional("Value", QValue, QuotedString(StringRef()));
    if (!IO.outputting())
      I.Value = QValue.Ref;

    QuotedString QExpr(I.ValueExpr);
    IO.mapOptional("Expr", QExpr, QuotedString(StringRef()));
    if (!IO.outputting())
      I.ValueExpr = QExpr.Ref;
  }
};

template <> struct MappingTraits<EnumInfo> {
  static void mapping(IO &IO, EnumInfo &I) {
    symbolInfoMapping(IO, I);
    IO.mapOptional("Scoped", I.Scoped, false);
    IO.mapOptional("BaseType", I.BaseType);
    IO.mapOptional("Members", I.Members);
  }
};

template <> struct MappingTraits<TypedefInfo> {
  static void mapping(IO &IO, TypedefInfo &I) {
    symbolInfoMapping(IO, I);
    IO.mapOptional("Underlying", I.Underlying.Type);
    IO.mapOptional("IsUsing", I.IsUsing, false);
  }
};

template <> struct MappingTraits<FunctionInfo> {
  static void mapping(IO &IO, FunctionInfo &I) {
    symbolInfoMapping(IO, I);
    IO.mapOptional("IsMethod", I.IsMethod, false);
    IO.mapOptional("Parent", I.Parent, Reference());
    IO.mapOptional("Params", I.Params);
    IO.mapOptional("ReturnType", I.ReturnType);
    // clang::AccessSpecifier::AS_none is used as the default here because it's
    // the AS that shouldn't be part of the output. Even though AS_public is the
    // default in the struct, it should be displayed in the YAML output.
    IO.mapOptional("Access", I.Access, clang::AccessSpecifier::AS_none);
    IO.mapOptional("Template", I.Template);
  }
};

template <> struct MappingTraits<TemplateParamInfo> {
  static void mapping(IO &IO, TemplateParamInfo &I) {
    QuotedString QContents(I.Contents);
    IO.mapOptional("Contents", QContents, QuotedString(StringRef()));
    if (!IO.outputting())
      I.Contents = QContents.Ref;
  }
};

template <> struct MappingTraits<TemplateSpecializationInfo> {
  static void mapping(IO &IO, TemplateSpecializationInfo &I) {
    IO.mapOptional("SpecializationOf", I.SpecializationOf);
    IO.mapOptional("Params", I.Params);
  }
};

template <> struct MappingTraits<TemplateInfo> {
  static void mapping(IO &IO, TemplateInfo &I) {
    IO.mapOptional("Params", I.Params);
    IO.mapOptional("Specialization", I.Specialization,
                   std::optional<TemplateSpecializationInfo>());
  }
};

template <> struct MappingTraits<CommentInfo> {
  static void mapping(IO &IO, CommentInfo &I) { commentInfoMapping(IO, I); }
};

} // end namespace yaml
} // end namespace llvm

namespace clang {
namespace doc {

/// Generator for YAML documentation.
class YAMLGenerator : public Generator {
public:
  static const char *Format;

  llvm::Error generateDocumentation(
      StringRef RootDir, llvm::StringMap<doc::OwnedPtr<doc::Info>> Infos,
      const ClangDocContext &CDCtx, std::string DirName) override;
  llvm::Error generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                 const ClangDocContext &CDCtx) override;
};

const char *YAMLGenerator::Format = "yaml";

llvm::Error YAMLGenerator::generateDocumentation(
    StringRef RootDir, llvm::StringMap<doc::OwnedPtr<doc::Info>> Infos,
    const ClangDocContext &CDCtx, std::string DirName) {
  for (const auto &Group : Infos) {
    doc::Info *Info = getPtr(Group.getValue());

    // Output file names according to the USR except the global namesapce.
    // Anonymous namespaces are taken care of in serialization, so here we can
    // safely assume an unnamed namespace is the global one.
    llvm::SmallString<128> Path;
    llvm::sys::path::native(RootDir, Path);
    if (Info->IT == InfoType::IT_namespace && Info->Name.empty()) {
      llvm::sys::path::append(Path, "index.yaml");
    } else {
      llvm::sys::path::append(Path, Group.getKey() + ".yaml");
    }

    std::error_code FileErr;
    llvm::raw_fd_ostream InfoOS(Path, FileErr, llvm::sys::fs::OF_Text);
    if (FileErr) {
      return llvm::createStringError(FileErr, "Error opening file '%s'",
                                     Path.c_str());
    }

    if (llvm::Error Err = generateDocForInfo(Info, InfoOS, CDCtx)) {
      return Err;
    }
  }

  return llvm::Error::success();
}

llvm::Error YAMLGenerator::generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                              const ClangDocContext &CDCtx) {
  llvm::yaml::Output InfoYAML(OS);
  switch (I->IT) {
  case InfoType::IT_namespace:
    InfoYAML << *static_cast<clang::doc::NamespaceInfo *>(I);
    break;
  case InfoType::IT_record:
    InfoYAML << *static_cast<clang::doc::RecordInfo *>(I);
    break;
  case InfoType::IT_enum:
    InfoYAML << *static_cast<clang::doc::EnumInfo *>(I);
    break;
  case InfoType::IT_function:
    InfoYAML << *static_cast<clang::doc::FunctionInfo *>(I);
    break;
  case InfoType::IT_typedef:
    InfoYAML << *static_cast<clang::doc::TypedefInfo *>(I);
    break;
  case InfoType::IT_concept:
  case InfoType::IT_variable:
  case InfoType::IT_friend:
    break;
  case InfoType::IT_default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unexpected InfoType");
  }
  return llvm::Error::success();
}

static GeneratorRegistry::Add<YAMLGenerator> YAML(YAMLGenerator::Format,
                                                  "Generator for YAML output.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the generator.
volatile int YAMLGeneratorAnchorSource = 0;

} // namespace doc
} // namespace clang
