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
LLVM_YAML_IS_SEQUENCE_VECTOR(std::unique_ptr<CommentInfo>)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::SmallString<16>)

namespace llvm {
namespace yaml {

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

// Scalars to YAML output.
template <unsigned U> struct ScalarTraits<SmallString<U>> {

  static void output(const SmallString<U> &S, void *, llvm::raw_ostream &OS) {
    for (const auto &C : S)
      OS << C;
  }

  static StringRef input(StringRef Scalar, void *, SmallString<U> &Value) {
    Value.assign(Scalar.begin(), Scalar.end());
    return StringRef();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Single; }
};

template <> struct ScalarTraits<std::array<unsigned char, 20>> {

  static void output(const std::array<unsigned char, 20> &S, void *,
                     llvm::raw_ostream &OS) {
    OS << toHex(toStringRef(S));
  }

  static StringRef input(StringRef Scalar, void *,
                         std::array<unsigned char, 20> &Value) {
    if (Scalar.size() != 40)
      return "Error: Incorrect scalar size for USR.";
    Value = StringToSymbol(Scalar);
    return StringRef();
  }

  static SymbolID StringToSymbol(llvm::StringRef Value) {
    SymbolID USR;
    std::string HexString = fromHex(Value);
    std::copy(HexString.begin(), HexString.end(), USR.begin());
    return SymbolID(USR);
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Single; }
};

// Helper functions to map infos to YAML.

static void TypeInfoMapping(IO &IO, TypeInfo &I) {
  IO.mapOptional("Type", I.Type, Reference());
}

static void FieldTypeInfoMapping(IO &IO, FieldTypeInfo &I) {
  TypeInfoMapping(IO, I);
  IO.mapOptional("Name", I.Name, SmallString<16>());
  IO.mapOptional("DefaultValue", I.DefaultValue, SmallString<16>());
}

static void InfoMapping(IO &IO, Info &I) {
  IO.mapRequired("USR", I.USR);
  IO.mapOptional("Name", I.Name, SmallString<16>());
  IO.mapOptional("Path", I.Path, SmallString<128>());
  IO.mapOptional("Namespace", I.Namespace, llvm::SmallVector<Reference, 4>());
  IO.mapOptional("Description", I.Description);
}

static void SymbolInfoMapping(IO &IO, SymbolInfo &I) {
  InfoMapping(IO, I);
  IO.mapOptional("DefLocation", I.DefLoc, std::optional<Location>());
  IO.mapOptional("Location", I.Loc, llvm::SmallVector<Location, 2>());
}

static void RecordInfoMapping(IO &IO, RecordInfo &I) {
  SymbolInfoMapping(IO, I);
  IO.mapOptional("TagType", I.TagType);
  IO.mapOptional("IsTypeDef", I.IsTypeDef, false);
  IO.mapOptional("Members", I.Members);
  IO.mapOptional("Bases", I.Bases);
  IO.mapOptional("Parents", I.Parents, llvm::SmallVector<Reference, 4>());
  IO.mapOptional("VirtualParents", I.VirtualParents,
                 llvm::SmallVector<Reference, 4>());
  IO.mapOptional("ChildRecords", I.Children.Records, std::vector<Reference>());
  IO.mapOptional("ChildFunctions", I.Children.Functions);
  IO.mapOptional("ChildEnums", I.Children.Enums);
  IO.mapOptional("ChildTypedefs", I.Children.Typedefs);
  IO.mapOptional("Template", I.Template);
}

static void CommentInfoMapping(IO &IO, CommentInfo &I) {
  IO.mapOptional("Kind", I.Kind, SmallString<16>());
  IO.mapOptional("Text", I.Text, SmallString<64>());
  IO.mapOptional("Name", I.Name, SmallString<16>());
  IO.mapOptional("Direction", I.Direction, SmallString<8>());
  IO.mapOptional("ParamName", I.ParamName, SmallString<16>());
  IO.mapOptional("CloseName", I.CloseName, SmallString<16>());
  IO.mapOptional("SelfClosing", I.SelfClosing, false);
  IO.mapOptional("Explicit", I.Explicit, false);
  IO.mapOptional("Args", I.Args, llvm::SmallVector<SmallString<16>, 4>());
  IO.mapOptional("AttrKeys", I.AttrKeys,
                 llvm::SmallVector<SmallString<16>, 4>());
  IO.mapOptional("AttrValues", I.AttrValues,
                 llvm::SmallVector<SmallString<16>, 4>());
  IO.mapOptional("Children", I.Children);
}

// Template specialization to YAML traits for Infos.

template <> struct MappingTraits<Location> {
  static void mapping(IO &IO, Location &Loc) {
    IO.mapOptional("LineNumber", Loc.LineNumber, 0);
    IO.mapOptional("Filename", Loc.Filename, SmallString<32>());
  }
};

template <> struct MappingTraits<Reference> {
  static void mapping(IO &IO, Reference &Ref) {
    IO.mapOptional("Type", Ref.RefType, InfoType::IT_default);
    IO.mapOptional("Name", Ref.Name, SmallString<16>());
    IO.mapOptional("QualName", Ref.QualName, SmallString<16>());
    IO.mapOptional("USR", Ref.USR, SymbolID());
    IO.mapOptional("Path", Ref.Path, SmallString<128>());
  }
};

template <> struct MappingTraits<TypeInfo> {
  static void mapping(IO &IO, TypeInfo &I) { TypeInfoMapping(IO, I); }
};

template <> struct MappingTraits<FieldTypeInfo> {
  static void mapping(IO &IO, FieldTypeInfo &I) {
    TypeInfoMapping(IO, I);
    IO.mapOptional("Name", I.Name, SmallString<16>());
    IO.mapOptional("DefaultValue", I.DefaultValue, SmallString<16>());
  }
};

template <> struct MappingTraits<MemberTypeInfo> {
  static void mapping(IO &IO, MemberTypeInfo &I) {
    FieldTypeInfoMapping(IO, I);
    // clang::AccessSpecifier::AS_none is used as the default here because it's
    // the AS that shouldn't be part of the output. Even though AS_public is the
    // default in the struct, it should be displayed in the YAML output.
    IO.mapOptional("Access", I.Access, clang::AccessSpecifier::AS_none);
    IO.mapOptional("Description", I.Description);
  }
};

template <> struct MappingTraits<NamespaceInfo> {
  static void mapping(IO &IO, NamespaceInfo &I) {
    InfoMapping(IO, I);
    IO.mapOptional("ChildNamespaces", I.Children.Namespaces,
                   std::vector<Reference>());
    IO.mapOptional("ChildRecords", I.Children.Records,
                   std::vector<Reference>());
    IO.mapOptional("ChildFunctions", I.Children.Functions);
    IO.mapOptional("ChildEnums", I.Children.Enums);
    IO.mapOptional("ChildTypedefs", I.Children.Typedefs);
  }
};

template <> struct MappingTraits<RecordInfo> {
  static void mapping(IO &IO, RecordInfo &I) { RecordInfoMapping(IO, I); }
};

template <> struct MappingTraits<BaseRecordInfo> {
  static void mapping(IO &IO, BaseRecordInfo &I) {
    RecordInfoMapping(IO, I);
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
    IO.mapOptional("Name", I.Name);
    IO.mapOptional("Value", I.Value);
    IO.mapOptional("Expr", I.ValueExpr, SmallString<16>());
  }
};

template <> struct MappingTraits<EnumInfo> {
  static void mapping(IO &IO, EnumInfo &I) {
    SymbolInfoMapping(IO, I);
    IO.mapOptional("Scoped", I.Scoped, false);
    IO.mapOptional("BaseType", I.BaseType);
    IO.mapOptional("Members", I.Members);
  }
};

template <> struct MappingTraits<TypedefInfo> {
  static void mapping(IO &IO, TypedefInfo &I) {
    SymbolInfoMapping(IO, I);
    IO.mapOptional("Underlying", I.Underlying.Type);
    IO.mapOptional("IsUsing", I.IsUsing, false);
  }
};

template <> struct MappingTraits<FunctionInfo> {
  static void mapping(IO &IO, FunctionInfo &I) {
    SymbolInfoMapping(IO, I);
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
    IO.mapOptional("Contents", I.Contents);
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
  static void mapping(IO &IO, CommentInfo &I) { CommentInfoMapping(IO, I); }
};

template <> struct MappingTraits<std::unique_ptr<CommentInfo>> {
  static void mapping(IO &IO, std::unique_ptr<CommentInfo> &I) {
    if (I)
      CommentInfoMapping(IO, *I);
  }
};

} // end namespace yaml
} // end namespace llvm

namespace clang {
namespace doc {

/// Generator for YAML documentation.
class YAMLGenerator : public Generator {
public:
  static const char *Format;

  llvm::Error generateDocs(StringRef RootDir,
                           llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
                           const ClangDocContext &CDCtx) override;
  llvm::Error generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                 const ClangDocContext &CDCtx) override;
};

const char *YAMLGenerator::Format = "yaml";

llvm::Error
YAMLGenerator::generateDocs(StringRef RootDir,
                            llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
                            const ClangDocContext &CDCtx) {
  for (const auto &Group : Infos) {
    doc::Info *Info = Group.getValue().get();

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
