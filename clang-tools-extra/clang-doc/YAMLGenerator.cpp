//===--  ClangDocYAML.cpp - ClangDoc YAML -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Implementation of the YAML generator, converting decl info into YAML output.
//===----------------------------------------------------------------------===//

#include "Generators.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::doc;

LLVM_YAML_IS_SEQUENCE_VECTOR(FieldTypeInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(MemberTypeInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(Reference)
LLVM_YAML_IS_SEQUENCE_VECTOR(Location)
LLVM_YAML_IS_SEQUENCE_VECTOR(CommentInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(FunctionInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(EnumInfo)
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
    IO.enumCase(Value, "Struct", clang::TagTypeKind::TTK_Struct);
    IO.enumCase(Value, "Interface", clang::TagTypeKind::TTK_Interface);
    IO.enumCase(Value, "Union", clang::TagTypeKind::TTK_Union);
    IO.enumCase(Value, "Class", clang::TagTypeKind::TTK_Class);
    IO.enumCase(Value, "Enum", clang::TagTypeKind::TTK_Enum);
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
}

static void InfoMapping(IO &IO, Info &I) {
  IO.mapRequired("USR", I.USR);
  IO.mapOptional("Name", I.Name, SmallString<16>());
  IO.mapOptional("Namespace", I.Namespace, llvm::SmallVector<Reference, 4>());
  IO.mapOptional("Description", I.Description);
}

static void SymbolInfoMapping(IO &IO, SymbolInfo &I) {
  InfoMapping(IO, I);
  IO.mapOptional("DefLocation", I.DefLoc, Optional<Location>());
  IO.mapOptional("Location", I.Loc, llvm::SmallVector<Location, 2>());
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
    IO.mapOptional("USR", Ref.USR, SymbolID());
  }
};

template <> struct MappingTraits<TypeInfo> {
  static void mapping(IO &IO, TypeInfo &I) { TypeInfoMapping(IO, I); }
};

template <> struct MappingTraits<FieldTypeInfo> {
  static void mapping(IO &IO, FieldTypeInfo &I) {
    TypeInfoMapping(IO, I);
    IO.mapOptional("Name", I.Name, SmallString<16>());
  }
};

template <> struct MappingTraits<MemberTypeInfo> {
  static void mapping(IO &IO, MemberTypeInfo &I) {
    FieldTypeInfoMapping(IO, I);
    IO.mapOptional("Access", I.Access, clang::AccessSpecifier::AS_none);
  }
};

template <> struct MappingTraits<NamespaceInfo> {
  static void mapping(IO &IO, NamespaceInfo &I) {
    InfoMapping(IO, I);
    IO.mapOptional("ChildNamespaces", I.ChildNamespaces,
                   std::vector<Reference>());
    IO.mapOptional("ChildRecords", I.ChildRecords, std::vector<Reference>());
    IO.mapOptional("ChildFunctions", I.ChildFunctions);
    IO.mapOptional("ChildEnums", I.ChildEnums);
  }
};

template <> struct MappingTraits<RecordInfo> {
  static void mapping(IO &IO, RecordInfo &I) {
    SymbolInfoMapping(IO, I);
    IO.mapOptional("TagType", I.TagType, clang::TagTypeKind::TTK_Struct);
    IO.mapOptional("Members", I.Members);
    IO.mapOptional("Parents", I.Parents, llvm::SmallVector<Reference, 4>());
    IO.mapOptional("VirtualParents", I.VirtualParents,
                   llvm::SmallVector<Reference, 4>());
    IO.mapOptional("ChildRecords", I.ChildRecords, std::vector<Reference>());
    IO.mapOptional("ChildFunctions", I.ChildFunctions);
    IO.mapOptional("ChildEnums", I.ChildEnums);
  }
};

template <> struct MappingTraits<EnumInfo> {
  static void mapping(IO &IO, EnumInfo &I) {
    SymbolInfoMapping(IO, I);
    IO.mapOptional("Scoped", I.Scoped, false);
    IO.mapOptional("Members", I.Members);
  }
};

template <> struct MappingTraits<FunctionInfo> {
  static void mapping(IO &IO, FunctionInfo &I) {
    SymbolInfoMapping(IO, I);
    IO.mapOptional("IsMethod", I.IsMethod, false);
    IO.mapOptional("Parent", I.Parent, Reference());
    IO.mapOptional("Params", I.Params);
    IO.mapOptional("ReturnType", I.ReturnType);
    IO.mapOptional("Access", I.Access, clang::AccessSpecifier::AS_none);
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

  bool generateDocForInfo(Info *I, llvm::raw_ostream &OS) override;
};

const char *YAMLGenerator::Format = "yaml";

bool YAMLGenerator::generateDocForInfo(Info *I, llvm::raw_ostream &OS) {
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
  case InfoType::IT_default:
    llvm::errs() << "Unexpected info type in index.\n";
    return true;
  }
  return false;
}

static GeneratorRegistry::Add<YAMLGenerator> YAML(YAMLGenerator::Format,
                                                  "Generator for YAML output.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the generator.
volatile int YAMLGeneratorAnchorSource = 0;

} // namespace doc
} // namespace clang
