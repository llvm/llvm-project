//===-- APINotesYAMLCompiler.cpp - API Notes YAML Format Reader -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The types defined locally are designed to represent the YAML state, which
// adds an additional bit of state: e.g. a tri-state boolean attribute (yes, no,
// not applied) becomes a tri-state boolean + present.  As a result, while these
// enumerations appear to be redefining constants from the attributes table
// data, they are distinct.
//

#include "clang/APINotes/APINotesYAMLCompiler.h"
#include "clang/APINotes/Types.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include <optional>
#include <vector>
using namespace clang;
using namespace api_notes;

namespace {
enum class APIAvailability {
  Available = 0,
  None,
  NonSwift,
};
} // namespace

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<APIAvailability> {
  static void enumeration(IO &IO, APIAvailability &AA) {
    IO.enumCase(AA, "none", APIAvailability::None);
    IO.enumCase(AA, "nonswift", APIAvailability::NonSwift);
    IO.enumCase(AA, "available", APIAvailability::Available);
  }
};
} // namespace yaml
} // namespace llvm

namespace {
enum class MethodKind {
  Class,
  Instance,
};
} // namespace

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<MethodKind> {
  static void enumeration(IO &IO, MethodKind &MK) {
    IO.enumCase(MK, "Class", MethodKind::Class);
    IO.enumCase(MK, "Instance", MethodKind::Instance);
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct Param {
  unsigned Position;
  std::optional<bool> NoEscape = false;
  std::optional<NullabilityKind> Nullability;
  std::optional<RetainCountConventionKind> RetainCountConvention;
  StringRef Type;
};

typedef std::vector<Param> ParamsSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(Param)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(NullabilityKind)

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<NullabilityKind> {
  static void enumeration(IO &IO, NullabilityKind &NK) {
    IO.enumCase(NK, "Nonnull", NullabilityKind::NonNull);
    IO.enumCase(NK, "Optional", NullabilityKind::Nullable);
    IO.enumCase(NK, "Unspecified", NullabilityKind::Unspecified);
    IO.enumCase(NK, "NullableResult", NullabilityKind::NullableResult);
    // TODO: Mapping this to it's own value would allow for better cross
    // checking. Also the default should be Unknown.
    IO.enumCase(NK, "Scalar", NullabilityKind::Unspecified);

    // Aliases for compatibility with existing APINotes.
    IO.enumCase(NK, "N", NullabilityKind::NonNull);
    IO.enumCase(NK, "O", NullabilityKind::Nullable);
    IO.enumCase(NK, "U", NullabilityKind::Unspecified);
    IO.enumCase(NK, "S", NullabilityKind::Unspecified);
  }
};

template <> struct ScalarEnumerationTraits<RetainCountConventionKind> {
  static void enumeration(IO &IO, RetainCountConventionKind &RCCK) {
    IO.enumCase(RCCK, "none", RetainCountConventionKind::None);
    IO.enumCase(RCCK, "CFReturnsRetained",
                RetainCountConventionKind::CFReturnsRetained);
    IO.enumCase(RCCK, "CFReturnsNotRetained",
                RetainCountConventionKind::CFReturnsNotRetained);
    IO.enumCase(RCCK, "NSReturnsRetained",
                RetainCountConventionKind::NSReturnsRetained);
    IO.enumCase(RCCK, "NSReturnsNotRetained",
                RetainCountConventionKind::NSReturnsNotRetained);
  }
};

template <> struct MappingTraits<Param> {
  static void mapping(IO &IO, Param &P) {
    IO.mapRequired("Position", P.Position);
    IO.mapOptional("Nullability", P.Nullability, std::nullopt);
    IO.mapOptional("RetainCountConvention", P.RetainCountConvention);
    IO.mapOptional("NoEscape", P.NoEscape);
    IO.mapOptional("Type", P.Type, StringRef(""));
  }
};
} // namespace yaml
} // namespace llvm

namespace {
typedef std::vector<NullabilityKind> NullabilitySeq;

struct AvailabilityItem {
  APIAvailability Mode = APIAvailability::Available;
  StringRef Msg;
};

/// Old attribute deprecated in favor of SwiftName.
enum class FactoryAsInitKind {
  /// Infer based on name and type (the default).
  Infer,
  /// Treat as a class method.
  AsClassMethod,
  /// Treat as an initializer.
  AsInitializer,
};

struct Method {
  StringRef Selector;
  MethodKind Kind;
  ParamsSeq Params;
  NullabilitySeq Nullability;
  std::optional<NullabilityKind> NullabilityOfRet;
  std::optional<RetainCountConventionKind> RetainCountConvention;
  AvailabilityItem Availability;
  std::optional<bool> SwiftPrivate;
  StringRef SwiftName;
  FactoryAsInitKind FactoryAsInit = FactoryAsInitKind::Infer;
  bool DesignatedInit = false;
  bool Required = false;
  StringRef ResultType;
};

typedef std::vector<Method> MethodsSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(Method)

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<FactoryAsInitKind> {
  static void enumeration(IO &IO, FactoryAsInitKind &FIK) {
    IO.enumCase(FIK, "A", FactoryAsInitKind::Infer);
    IO.enumCase(FIK, "C", FactoryAsInitKind::AsClassMethod);
    IO.enumCase(FIK, "I", FactoryAsInitKind::AsInitializer);
  }
};

template <> struct MappingTraits<Method> {
  static void mapping(IO &IO, Method &M) {
    IO.mapRequired("Selector", M.Selector);
    IO.mapRequired("MethodKind", M.Kind);
    IO.mapOptional("Parameters", M.Params);
    IO.mapOptional("Nullability", M.Nullability);
    IO.mapOptional("NullabilityOfRet", M.NullabilityOfRet, std::nullopt);
    IO.mapOptional("RetainCountConvention", M.RetainCountConvention);
    IO.mapOptional("Availability", M.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", M.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftPrivate", M.SwiftPrivate);
    IO.mapOptional("SwiftName", M.SwiftName, StringRef(""));
    IO.mapOptional("FactoryAsInit", M.FactoryAsInit, FactoryAsInitKind::Infer);
    IO.mapOptional("DesignatedInit", M.DesignatedInit, false);
    IO.mapOptional("Required", M.Required, false);
    IO.mapOptional("ResultType", M.ResultType, StringRef(""));
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct Property {
  StringRef Name;
  std::optional<MethodKind> Kind;
  std::optional<NullabilityKind> Nullability;
  AvailabilityItem Availability;
  std::optional<bool> SwiftPrivate;
  StringRef SwiftName;
  std::optional<bool> SwiftImportAsAccessors;
  StringRef Type;
};

typedef std::vector<Property> PropertiesSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(Property)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<Property> {
  static void mapping(IO &IO, Property &P) {
    IO.mapRequired("Name", P.Name);
    IO.mapOptional("PropertyKind", P.Kind);
    IO.mapOptional("Nullability", P.Nullability, std::nullopt);
    IO.mapOptional("Availability", P.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", P.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftPrivate", P.SwiftPrivate);
    IO.mapOptional("SwiftName", P.SwiftName, StringRef(""));
    IO.mapOptional("SwiftImportAsAccessors", P.SwiftImportAsAccessors);
    IO.mapOptional("Type", P.Type, StringRef(""));
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct Class {
  StringRef Name;
  bool AuditedForNullability = false;
  AvailabilityItem Availability;
  std::optional<bool> SwiftPrivate;
  StringRef SwiftName;
  std::optional<StringRef> SwiftBridge;
  std::optional<StringRef> NSErrorDomain;
  std::optional<bool> SwiftImportAsNonGeneric;
  std::optional<bool> SwiftObjCMembers;
  MethodsSeq Methods;
  PropertiesSeq Properties;
};

typedef std::vector<Class> ClassesSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(Class)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<Class> {
  static void mapping(IO &IO, Class &C) {
    IO.mapRequired("Name", C.Name);
    IO.mapOptional("AuditedForNullability", C.AuditedForNullability, false);
    IO.mapOptional("Availability", C.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", C.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftPrivate", C.SwiftPrivate);
    IO.mapOptional("SwiftName", C.SwiftName, StringRef(""));
    IO.mapOptional("SwiftBridge", C.SwiftBridge);
    IO.mapOptional("NSErrorDomain", C.NSErrorDomain);
    IO.mapOptional("SwiftImportAsNonGeneric", C.SwiftImportAsNonGeneric);
    IO.mapOptional("SwiftObjCMembers", C.SwiftObjCMembers);
    IO.mapOptional("Methods", C.Methods);
    IO.mapOptional("Properties", C.Properties);
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct Function {
  StringRef Name;
  ParamsSeq Params;
  NullabilitySeq Nullability;
  std::optional<NullabilityKind> NullabilityOfRet;
  std::optional<api_notes::RetainCountConventionKind> RetainCountConvention;
  AvailabilityItem Availability;
  std::optional<bool> SwiftPrivate;
  StringRef SwiftName;
  StringRef Type;
  StringRef ResultType;
};

typedef std::vector<Function> FunctionsSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(Function)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<Function> {
  static void mapping(IO &IO, Function &F) {
    IO.mapRequired("Name", F.Name);
    IO.mapOptional("Parameters", F.Params);
    IO.mapOptional("Nullability", F.Nullability);
    IO.mapOptional("NullabilityOfRet", F.NullabilityOfRet, std::nullopt);
    IO.mapOptional("RetainCountConvention", F.RetainCountConvention);
    IO.mapOptional("Availability", F.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", F.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftPrivate", F.SwiftPrivate);
    IO.mapOptional("SwiftName", F.SwiftName, StringRef(""));
    IO.mapOptional("ResultType", F.ResultType, StringRef(""));
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct GlobalVariable {
  StringRef Name;
  std::optional<NullabilityKind> Nullability;
  AvailabilityItem Availability;
  std::optional<bool> SwiftPrivate;
  StringRef SwiftName;
  StringRef Type;
};

typedef std::vector<GlobalVariable> GlobalVariablesSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(GlobalVariable)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<GlobalVariable> {
  static void mapping(IO &IO, GlobalVariable &GV) {
    IO.mapRequired("Name", GV.Name);
    IO.mapOptional("Nullability", GV.Nullability, std::nullopt);
    IO.mapOptional("Availability", GV.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", GV.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftPrivate", GV.SwiftPrivate);
    IO.mapOptional("SwiftName", GV.SwiftName, StringRef(""));
    IO.mapOptional("Type", GV.Type, StringRef(""));
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct EnumConstant {
  StringRef Name;
  AvailabilityItem Availability;
  std::optional<bool> SwiftPrivate;
  StringRef SwiftName;
};

typedef std::vector<EnumConstant> EnumConstantsSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(EnumConstant)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<EnumConstant> {
  static void mapping(IO &IO, EnumConstant &EC) {
    IO.mapRequired("Name", EC.Name);
    IO.mapOptional("Availability", EC.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", EC.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftPrivate", EC.SwiftPrivate);
    IO.mapOptional("SwiftName", EC.SwiftName, StringRef(""));
  }
};
} // namespace yaml
} // namespace llvm

namespace {
/// Syntactic sugar for EnumExtensibility and FlagEnum
enum class EnumConvenienceAliasKind {
  /// EnumExtensibility: none, FlagEnum: false
  None,
  /// EnumExtensibility: open, FlagEnum: false
  CFEnum,
  /// EnumExtensibility: open, FlagEnum: true
  CFOptions,
  /// EnumExtensibility: closed, FlagEnum: false
  CFClosedEnum
};
} // namespace

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<EnumConvenienceAliasKind> {
  static void enumeration(IO &IO, EnumConvenienceAliasKind &ECAK) {
    IO.enumCase(ECAK, "none", EnumConvenienceAliasKind::None);
    IO.enumCase(ECAK, "CFEnum", EnumConvenienceAliasKind::CFEnum);
    IO.enumCase(ECAK, "NSEnum", EnumConvenienceAliasKind::CFEnum);
    IO.enumCase(ECAK, "CFOptions", EnumConvenienceAliasKind::CFOptions);
    IO.enumCase(ECAK, "NSOptions", EnumConvenienceAliasKind::CFOptions);
    IO.enumCase(ECAK, "CFClosedEnum", EnumConvenienceAliasKind::CFClosedEnum);
    IO.enumCase(ECAK, "NSClosedEnum", EnumConvenienceAliasKind::CFClosedEnum);
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct Tag {
  StringRef Name;
  AvailabilityItem Availability;
  StringRef SwiftName;
  std::optional<bool> SwiftPrivate;
  std::optional<StringRef> SwiftBridge;
  std::optional<StringRef> NSErrorDomain;
  std::optional<EnumExtensibilityKind> EnumExtensibility;
  std::optional<bool> FlagEnum;
  std::optional<EnumConvenienceAliasKind> EnumConvenienceKind;
};

typedef std::vector<Tag> TagsSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(Tag)

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<EnumExtensibilityKind> {
  static void enumeration(IO &IO, EnumExtensibilityKind &EEK) {
    IO.enumCase(EEK, "none", EnumExtensibilityKind::None);
    IO.enumCase(EEK, "open", EnumExtensibilityKind::Open);
    IO.enumCase(EEK, "closed", EnumExtensibilityKind::Closed);
  }
};

template <> struct MappingTraits<Tag> {
  static void mapping(IO &IO, Tag &T) {
    IO.mapRequired("Name", T.Name);
    IO.mapOptional("Availability", T.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", T.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftPrivate", T.SwiftPrivate);
    IO.mapOptional("SwiftName", T.SwiftName, StringRef(""));
    IO.mapOptional("SwiftBridge", T.SwiftBridge);
    IO.mapOptional("NSErrorDomain", T.NSErrorDomain);
    IO.mapOptional("EnumExtensibility", T.EnumExtensibility);
    IO.mapOptional("FlagEnum", T.FlagEnum);
    IO.mapOptional("EnumKind", T.EnumConvenienceKind);
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct Typedef {
  StringRef Name;
  AvailabilityItem Availability;
  StringRef SwiftName;
  std::optional<bool> SwiftPrivate;
  std::optional<StringRef> SwiftBridge;
  std::optional<StringRef> NSErrorDomain;
  std::optional<SwiftNewTypeKind> SwiftType;
};

typedef std::vector<Typedef> TypedefsSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(Typedef)

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<SwiftNewTypeKind> {
  static void enumeration(IO &IO, SwiftNewTypeKind &SWK) {
    IO.enumCase(SWK, "none", SwiftNewTypeKind::None);
    IO.enumCase(SWK, "struct", SwiftNewTypeKind::Struct);
    IO.enumCase(SWK, "enum", SwiftNewTypeKind::Enum);
  }
};

template <> struct MappingTraits<Typedef> {
  static void mapping(IO &IO, Typedef &T) {
    IO.mapRequired("Name", T.Name);
    IO.mapOptional("Availability", T.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", T.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftPrivate", T.SwiftPrivate);
    IO.mapOptional("SwiftName", T.SwiftName, StringRef(""));
    IO.mapOptional("SwiftBridge", T.SwiftBridge);
    IO.mapOptional("NSErrorDomain", T.NSErrorDomain);
    IO.mapOptional("SwiftWrapper", T.SwiftType);
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct Namespace;
typedef std::vector<Namespace> NamespacesSeq;

struct TopLevelItems {
  ClassesSeq Classes;
  ClassesSeq Protocols;
  FunctionsSeq Functions;
  GlobalVariablesSeq Globals;
  EnumConstantsSeq EnumConstants;
  TagsSeq Tags;
  TypedefsSeq Typedefs;
  NamespacesSeq Namespaces;
};
} // namespace

namespace llvm {
namespace yaml {
static void mapTopLevelItems(IO &IO, TopLevelItems &TLI) {
  IO.mapOptional("Classes", TLI.Classes);
  IO.mapOptional("Protocols", TLI.Protocols);
  IO.mapOptional("Functions", TLI.Functions);
  IO.mapOptional("Globals", TLI.Globals);
  IO.mapOptional("Enumerators", TLI.EnumConstants);
  IO.mapOptional("Tags", TLI.Tags);
  IO.mapOptional("Typedefs", TLI.Typedefs);
  IO.mapOptional("Namespaces", TLI.Namespaces);
}
} // namespace yaml
} // namespace llvm

namespace {
struct Namespace {
  StringRef Name;
  AvailabilityItem Availability;
  StringRef SwiftName;
  std::optional<bool> SwiftPrivate;
  TopLevelItems Items;
};
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(Namespace)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<Namespace> {
  static void mapping(IO &IO, Namespace &T) {
    IO.mapRequired("Name", T.Name);
    IO.mapOptional("Availability", T.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", T.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftPrivate", T.SwiftPrivate);
    IO.mapOptional("SwiftName", T.SwiftName, StringRef(""));
    mapTopLevelItems(IO, T.Items);
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct Versioned {
  VersionTuple Version;
  TopLevelItems Items;
};

typedef std::vector<Versioned> VersionedSeq;
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(Versioned)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<Versioned> {
  static void mapping(IO &IO, Versioned &V) {
    IO.mapRequired("Version", V.Version);
    mapTopLevelItems(IO, V.Items);
  }
};
} // namespace yaml
} // namespace llvm

namespace {
struct Module {
  StringRef Name;
  AvailabilityItem Availability;
  TopLevelItems TopLevel;
  VersionedSeq SwiftVersions;

  std::optional<bool> SwiftInferImportAsMember;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() /*const*/;
#endif
};
} // namespace

namespace llvm {
namespace yaml {
template <> struct MappingTraits<Module> {
  static void mapping(IO &IO, Module &M) {
    IO.mapRequired("Name", M.Name);
    IO.mapOptional("Availability", M.Availability.Mode,
                   APIAvailability::Available);
    IO.mapOptional("AvailabilityMsg", M.Availability.Msg, StringRef(""));
    IO.mapOptional("SwiftInferImportAsMember", M.SwiftInferImportAsMember);
    mapTopLevelItems(IO, M.TopLevel);
    IO.mapOptional("SwiftVersions", M.SwiftVersions);
  }
};
} // namespace yaml
} // namespace llvm

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void Module::dump() {
  llvm::yaml::Output OS(llvm::errs());
  OS << *this;
}
#endif

namespace {
bool parseAPINotes(StringRef YI, Module &M, llvm::SourceMgr::DiagHandlerTy Diag,
                   void *DiagContext) {
  llvm::yaml::Input IS(YI, nullptr, Diag, DiagContext);
  IS >> M;
  return static_cast<bool>(IS.error());
}
} // namespace

bool clang::api_notes::parseAndDumpAPINotes(StringRef YI,
                                            llvm::raw_ostream &OS) {
  Module M;
  if (parseAPINotes(YI, M, nullptr, nullptr))
    return true;

  llvm::yaml::Output YOS(OS);
  YOS << M;

  return false;
}
