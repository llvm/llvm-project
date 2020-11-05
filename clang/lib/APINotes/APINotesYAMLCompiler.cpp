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
#include "llvm/ADT/Optional.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include <vector>
using namespace clang;
using namespace api_notes;

/*
 
 YAML Format specification.

 Nullability should be expressed using one of the following values:
   O - Optional (or Nullable)
   N - Not Optional
   S - Scalar
   U - Unknown
 Note, the API is considered 'audited' when at least the return value or a
 parameter has a nullability value. For 'audited' APIs, we assume the default
 nullability for any underspecified type.

---
 Name: AppKit             # The name of the framework

 Availability: OSX        # Optional: Specifies which platform the API is
                          # available on. [OSX / iOS / none/
                          #                available / nonswift]

 AvailabilityMsg: ""  # Optional: Custom availability message to display to
                          # the user, when API is not available.

 Classes:                 # List of classes
 ...
 Protocols:               # List of protocols
 ...
 Functions:               # List of functions
 ...
 Globals:                 # List of globals
 ...
 Enumerators:             # List of enumerators
 ...
 Tags:                    # List of tags (struct/union/enum/C++ class)
 ...
 Typedefs:                # List of typedef-names and C++11 type aliases
 ...

 Each class and protocol is defined as following:

 - Name: NSView                       # The name of the class
 
   AuditedForNullability: false       # Optional: Specifies if the whole class
                                      # has been audited for nullability.
                                      # If yes, we assume all the methods and
                                      # properties of the class have default
                                      # nullability unless it is overwritten by
                                      # a method/property specific info below.
                                      # This applies to all classes, extensions,
                                      # and categories of the class defined in
                                      # the current framework/module.
                                      # (false/true)

   Availability: OSX

   AvailabilityMsg: ""

   Methods:
     - Selector: "setSubviews:"       # Full name

       MethodKind: Instance           # [Class/Instance]

       Nullability: [N, N, O, S]      # The nullability of parameters in
                                      # the signature.

       NullabilityOfRet: O            # The nullability of the return value.

       Availability: OSX

       AvailabilityMsg: ""

       DesignatedInit: false          # Optional: Specifies if this method is a
                                      # designated initializer (false/true)

       Required: false                # Optional: Specifies if this method is a
                                      # required initializer (false/true)

   Properties:
     - Name: window

       Nullability: O

       Availability: OSX

       AvailabilityMsg: ""

 The protocol definition format is the same as the class definition.

 Each function definition is of the following form:

 - Name: "myGlobalFunction"           # Full name

   Nullability: [N, N, O, S]          # The nullability of parameters in
                                      # the signature.

   NullabilityOfRet: O                # The nullability of the return value.

   Availability: OSX

   AvailabilityMsg: ""

Each global variable definition is of the following form:

 - Name: MyGlobalVar

   Nullability: O

   Availability: OSX

   AvailabilityMsg: ""

*/

namespace {
enum class APIAvailability {
  Available = 0,
  OSX,
  IOS,
  None,
  NonSwift,
};
} // namespace

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<APIAvailability> {
  static void enumeration(IO &IO, APIAvailability &AA) {
    IO.enumCase(AA, "OSX", APIAvailability::OSX);
    IO.enumCase(AA, "iOS", APIAvailability::IOS);
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
  Optional<bool> NoEscape = false;
  Optional<NullabilityKind> Nullability;
  Optional<RetainCountConventionKind> RetainCountConvention;
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
    IO.mapOptional("Nullability", P.Nullability, llvm::None);
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
  Optional<NullabilityKind> NullabilityOfRet;
  Optional<RetainCountConventionKind> RetainCountConvention;
  AvailabilityItem Availability;
  Optional<bool> SwiftPrivate;
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
    IO.mapOptional("NullabilityOfRet", M.NullabilityOfRet, llvm::None);
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
  llvm::Optional<MethodKind> Kind;
  llvm::Optional<NullabilityKind> Nullability;
  AvailabilityItem Availability;
  Optional<bool> SwiftPrivate;
  StringRef SwiftName;
  Optional<bool> SwiftImportAsAccessors;
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
    IO.mapOptional("Nullability", P.Nullability, llvm::None);
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
  Optional<bool> SwiftPrivate;
  StringRef SwiftName;
  Optional<StringRef> SwiftBridge;
  Optional<StringRef> NSErrorDomain;
  Optional<bool> SwiftImportAsNonGeneric;
  Optional<bool> SwiftObjCMembers;
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
  Optional<NullabilityKind> NullabilityOfRet;
  Optional<api_notes::RetainCountConventionKind> RetainCountConvention;
  AvailabilityItem Availability;
  Optional<bool> SwiftPrivate;
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
    IO.mapOptional("NullabilityOfRet", F.NullabilityOfRet, llvm::None);
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
  llvm::Optional<NullabilityKind> Nullability;
  AvailabilityItem Availability;
  Optional<bool> SwiftPrivate;
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
    IO.mapOptional("Nullability", GV.Nullability, llvm::None);
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
  Optional<bool> SwiftPrivate;
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
  Optional<bool> SwiftPrivate;
  Optional<StringRef> SwiftBridge;
  Optional<StringRef> NSErrorDomain;
  Optional<EnumExtensibilityKind> EnumExtensibility;
  Optional<bool> FlagEnum;
  Optional<EnumConvenienceAliasKind> EnumConvenienceKind;
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
  Optional<bool> SwiftPrivate;
  Optional<StringRef> SwiftBridge;
  Optional<StringRef> NSErrorDomain;
  Optional<SwiftNewTypeKind> SwiftType;
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
struct TopLevelItems {
  ClassesSeq Classes;
  ClassesSeq Protocols;
  FunctionsSeq Functions;
  GlobalVariablesSeq Globals;
  EnumConstantsSeq EnumConstants;
  TagsSeq Tags;
  TypedefsSeq Typedefs;
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
}
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

  llvm::Optional<bool> SwiftInferImportAsMember = {llvm::None};

  LLVM_DUMP_METHOD void dump() /*const*/;
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

void Module::dump() {
  llvm::yaml::Output OS(llvm::errs());
  OS << *this;
}

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

#include "clang/APINotes/APINotesReader.h"
#include "clang/APINotes/Types.h"
#include "clang/APINotes/APINotesWriter.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include <algorithm>

namespace {
  using namespace api_notes;

  class YAMLConverter {
    const Module &TheModule;
    const FileEntry *SourceFile;
    APINotesWriter *Writer;
    llvm::raw_ostream &OS;
    llvm::SourceMgr::DiagHandlerTy DiagHandler;
    void *DiagHandlerCtxt;
    bool ErrorOccured;

    /// Emit a diagnostic
    bool emitError(llvm::Twine message) {
      DiagHandler(llvm::SMDiagnostic("", llvm::SourceMgr::DK_Error,
                                     message.str()),
                  DiagHandlerCtxt);
      ErrorOccured = true;
      return true;
    }

  public:
    YAMLConverter(const Module &module,
                  const FileEntry *sourceFile,
                  llvm::raw_ostream &os,
                  llvm::SourceMgr::DiagHandlerTy diagHandler,
                  void *diagHandlerCtxt) :
      TheModule(module), SourceFile(sourceFile), Writer(0), OS(os),
      DiagHandler(diagHandler), DiagHandlerCtxt(diagHandlerCtxt),
      ErrorOccured(false) {}

    bool convertAvailability(const AvailabilityItem &in,
                             CommonEntityInfo &outInfo,
                             llvm::StringRef apiName) {
      // Populate the unavailability information.
      outInfo.Unavailable = (in.Mode == APIAvailability::None);
      outInfo.UnavailableInSwift = (in.Mode == APIAvailability::NonSwift);
      if (outInfo.Unavailable || outInfo.UnavailableInSwift) {
        outInfo.UnavailableMsg = std::string(in.Msg);
      } else {
        if (!in.Msg.empty()) {
          emitError("availability message for available API '" +
                    apiName + "' will not be used");
        }
      }
      return false;
    }

    void convertParams(const ParamsSeq &params, FunctionInfo &outInfo) {
      for (const auto &p : params) {
        ParamInfo pi;
        if (p.Nullability)
          pi.setNullabilityAudited(*p.Nullability);
        pi.setNoEscape(p.NoEscape);
        pi.setType(std::string(p.Type));
        pi.setRetainCountConvention(p.RetainCountConvention);
        while (outInfo.Params.size() <= p.Position) {
          outInfo.Params.push_back(ParamInfo());
        }
        outInfo.Params[p.Position] |= pi;
      }
    }

    void convertNullability(const NullabilitySeq &nullability,
                            Optional<NullabilityKind> nullabilityOfRet,
                            FunctionInfo &outInfo,
                            llvm::StringRef apiName) {
      if (nullability.size() > FunctionInfo::getMaxNullabilityIndex()) {
        emitError("nullability info for " + apiName + " does not fit");
        return;
      }

      bool audited = false;
      unsigned int idx = 1;
      for (auto i = nullability.begin(),
                e = nullability.end(); i != e; ++i, ++idx){
        outInfo.addTypeInfo(idx, *i);
        audited = true;
      }
      if (nullabilityOfRet) {
        outInfo.addTypeInfo(0, *nullabilityOfRet);
        audited = true;
      } else if (audited) {
        outInfo.addTypeInfo(0, NullabilityKind::NonNull);
      }
      if (audited) {
        outInfo.NullabilityAudited = audited;
        outInfo.NumAdjustedNullable = idx;
      }
    }

    /// Convert the common parts of an entity from YAML.
    template<typename T>
    bool convertCommon(const T& common, CommonEntityInfo &info,
                       StringRef apiName) {
      convertAvailability(common.Availability, info, apiName);
      info.setSwiftPrivate(common.SwiftPrivate);
      info.SwiftName = std::string(common.SwiftName);
      return false;
    }
    
    /// Convert the common parts of a type entity from YAML.
    template<typename T>
    bool convertCommonType(const T& common, CommonTypeInfo &info,
                           StringRef apiName) {
      if (convertCommon(common, info, apiName))
        return true;

      info.setSwiftBridge(common.SwiftBridge);
      info.setNSErrorDomain(common.NSErrorDomain);
      return false;
    }

    // Translate from Method into ObjCMethodInfo and write it out.
    void convertMethod(const Method &meth,
                       ContextID classID, StringRef className,
                       VersionTuple swiftVersion) {
      ObjCMethodInfo mInfo;

      if (convertCommon(meth, mInfo, meth.Selector))
        return;

      // Check if the selector ends with ':' to determine if it takes arguments.
      bool takesArguments = meth.Selector.endswith(":");

      // Split the selector into pieces.
      llvm::SmallVector<StringRef, 4> a;
      meth.Selector.split(a, ":", /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      if (!takesArguments && a.size() > 1 ) {
        emitError("selector " + meth.Selector + "is missing a ':' at the end");
        return;
      }

      // Construct ObjCSelectorRef.
      api_notes::ObjCSelectorRef selectorRef;
      selectorRef.NumPieces = !takesArguments ? 0 : a.size();
      selectorRef.Identifiers = a;

      // Translate the initializer info.
      mInfo.DesignatedInit = meth.DesignatedInit;
      mInfo.RequiredInit = meth.Required;
      if (meth.FactoryAsInit != FactoryAsInitKind::Infer) {
        emitError("'FactoryAsInit' is no longer valid; "
                  "use 'SwiftName' instead");
      }
      mInfo.ResultType = std::string(meth.ResultType);

      // Translate parameter information.
      convertParams(meth.Params, mInfo);

      // Translate nullability info.
      convertNullability(meth.Nullability, meth.NullabilityOfRet,
                         mInfo, meth.Selector);

      mInfo.setRetainCountConvention(meth.RetainCountConvention);

      // Write it.
      Writer->addObjCMethod(classID, selectorRef,
                            meth.Kind == MethodKind::Instance,
                            mInfo, swiftVersion);
    }

    void convertContext(const Class &cl, bool isClass,
                        VersionTuple swiftVersion) {
      // Write the class.
      ObjCContextInfo cInfo;

      if (convertCommonType(cl, cInfo, cl.Name))
        return;

      if (cl.AuditedForNullability)
        cInfo.setDefaultNullability(NullabilityKind::NonNull);
      if (cl.SwiftImportAsNonGeneric)
        cInfo.setSwiftImportAsNonGeneric(*cl.SwiftImportAsNonGeneric);
      if (cl.SwiftObjCMembers)
        cInfo.setSwiftObjCMembers(*cl.SwiftObjCMembers);

      ContextID clID = Writer->addObjCContext(cl.Name, isClass, cInfo,
                                              swiftVersion);

      // Write all methods.
      llvm::StringMap<std::pair<bool, bool>> knownMethods;
      for (const auto &method : cl.Methods) {
        // Check for duplicate method definitions.
        bool isInstanceMethod = method.Kind == MethodKind::Instance;
        bool &known = isInstanceMethod ? knownMethods[method.Selector].first
                                       : knownMethods[method.Selector].second;
        if (known) {
          emitError(llvm::Twine("duplicate definition of method '") +
                    (isInstanceMethod? "-" : "+") + "[" + cl.Name + " " +
                    method.Selector + "]'");
          continue;
        }
        known = true;

        convertMethod(method, clID, cl.Name, swiftVersion);
      }

      // Write all properties.
      llvm::StringSet<> knownInstanceProperties;
      llvm::StringSet<> knownClassProperties;
      for (const auto &prop : cl.Properties) {
        // Check for duplicate property definitions.
        if ((!prop.Kind || *prop.Kind == MethodKind::Instance) &&
            !knownInstanceProperties.insert(prop.Name).second) {
          emitError("duplicate definition of instance property '" + cl.Name +
                    "." + prop.Name + "'");
          continue;
        }

        if ((!prop.Kind || *prop.Kind == MethodKind::Class) &&
            !knownClassProperties.insert(prop.Name).second) {
          emitError("duplicate definition of class property '" + cl.Name + "." +
                    prop.Name + "'");
          continue;
        }

        // Translate from Property into ObjCPropertyInfo.
        ObjCPropertyInfo pInfo;
        convertAvailability(prop.Availability, pInfo, prop.Name);
        pInfo.setSwiftPrivate(prop.SwiftPrivate);
        pInfo.SwiftName = std::string(prop.SwiftName);
        if (prop.Nullability)
          pInfo.setNullabilityAudited(*prop.Nullability);
        if (prop.SwiftImportAsAccessors)
          pInfo.setSwiftImportAsAccessors(*prop.SwiftImportAsAccessors);
        pInfo.setType(std::string(prop.Type));
        if (prop.Kind) {
          Writer->addObjCProperty(clID, prop.Name,
                                  *prop.Kind == MethodKind::Instance, pInfo,
                                  swiftVersion);
        } else {
          // Add both instance and class properties with this name.
          Writer->addObjCProperty(clID, prop.Name, true, pInfo, swiftVersion);
          Writer->addObjCProperty(clID, prop.Name, false, pInfo, swiftVersion);
        }
      }
    }

    void convertTopLevelItems(const TopLevelItems &items,
                              VersionTuple swiftVersion) {
      // Write all classes.
      llvm::StringSet<> knownClasses;
      for (const auto &cl : items.Classes) {
        // Check for duplicate class definitions.
        if (!knownClasses.insert(cl.Name).second) {
          emitError("multiple definitions of class '" + cl.Name + "'");
          continue;
        }

        convertContext(cl, /*isClass*/ true, swiftVersion);
      }

      // Write all protocols.
      llvm::StringSet<> knownProtocols;
      for (const auto &pr : items.Protocols) {
        // Check for duplicate protocol definitions.
        if (!knownProtocols.insert(pr.Name).second) {
          emitError("multiple definitions of protocol '" + pr.Name + "'");
          continue;
        }

        convertContext(pr, /*isClass*/ false, swiftVersion);
      }

      // Write all global variables.
      llvm::StringSet<> knownGlobals;
      for (const auto &global : items.Globals) {
        // Check for duplicate global variables.
        if (!knownGlobals.insert(global.Name).second) {
          emitError("multiple definitions of global variable '" +
                    global.Name + "'");
          continue;
        }

        GlobalVariableInfo info;
        convertAvailability(global.Availability, info, global.Name);
        info.setSwiftPrivate(global.SwiftPrivate);
        info.SwiftName = std::string(global.SwiftName);
        if (global.Nullability)
          info.setNullabilityAudited(*global.Nullability);
        info.setType(std::string(global.Type));
        Writer->addGlobalVariable(global.Name, info, swiftVersion);
      }

      // Write all global functions.
      llvm::StringSet<> knownFunctions;
      for (const auto &function : items.Functions) {
        // Check for duplicate global functions.
        if (!knownFunctions.insert(function.Name).second) {
          emitError("multiple definitions of global function '" +
                    function.Name + "'");
          continue;
        }

        GlobalFunctionInfo info;
        convertAvailability(function.Availability, info, function.Name);
        info.setSwiftPrivate(function.SwiftPrivate);
        info.SwiftName = std::string(function.SwiftName);
        convertParams(function.Params, info);
        convertNullability(function.Nullability,
                           function.NullabilityOfRet,
                           info, function.Name);
        info.ResultType = std::string(function.ResultType);
        info.setRetainCountConvention(function.RetainCountConvention);
        Writer->addGlobalFunction(function.Name, info, swiftVersion);
      }

      // Write all enumerators.
      llvm::StringSet<> knownEnumConstants;
      for (const auto &enumConstant : items.EnumConstants) {
        // Check for duplicate enumerators
        if (!knownEnumConstants.insert(enumConstant.Name).second) {
          emitError("multiple definitions of enumerator '" +
                    enumConstant.Name + "'");
          continue;
        }

        EnumConstantInfo info;
        convertAvailability(enumConstant.Availability, info, enumConstant.Name);
        info.setSwiftPrivate(enumConstant.SwiftPrivate);
        info.SwiftName = std::string(enumConstant.SwiftName);
        Writer->addEnumConstant(enumConstant.Name, info, swiftVersion);
      }

      // Write all tags.
      llvm::StringSet<> knownTags;
      for (const auto &t : items.Tags) {
        // Check for duplicate tag definitions.
        if (!knownTags.insert(t.Name).second) {
          emitError("multiple definitions Of tag '" + t.Name + "'");
          continue;
        }

        TagInfo tagInfo;
        if (convertCommonType(t, tagInfo, t.Name))
          continue;

        if (t.EnumConvenienceKind) {
          if (t.EnumExtensibility) {
            emitError(llvm::Twine(
                "cannot mix EnumKind and EnumExtensibility (for ") + t.Name +
                ")");
            continue;
          }
          if (t.FlagEnum) {
            emitError(llvm::Twine("cannot mix EnumKind and FlagEnum (for ") +
                t.Name + ")");
            continue;
          }
          switch (t.EnumConvenienceKind.getValue()) {
          case EnumConvenienceAliasKind::None:
            tagInfo.EnumExtensibility = EnumExtensibilityKind::None;
            tagInfo.setFlagEnum(false);
            break;
          case EnumConvenienceAliasKind::CFEnum:
            tagInfo.EnumExtensibility = EnumExtensibilityKind::Open;
            tagInfo.setFlagEnum(false);
            break;
          case EnumConvenienceAliasKind::CFOptions:
            tagInfo.EnumExtensibility = EnumExtensibilityKind::Open;
            tagInfo.setFlagEnum(true);
            break;
          case EnumConvenienceAliasKind::CFClosedEnum:
            tagInfo.EnumExtensibility = EnumExtensibilityKind::Closed;
            tagInfo.setFlagEnum(false);
            break;
          }
        } else {
          tagInfo.EnumExtensibility = t.EnumExtensibility;
          tagInfo.setFlagEnum(t.FlagEnum);          
        }

        Writer->addTag(t.Name, tagInfo, swiftVersion);
      }

      // Write all typedefs.
      llvm::StringSet<> knownTypedefs;
      for (const auto &t : items.Typedefs) {
        // Check for duplicate typedef definitions.
        if (!knownTypedefs.insert(t.Name).second) {
          emitError("multiple definitions of typedef '" + t.Name + "'");
          continue;
        }

        TypedefInfo typedefInfo;
        if (convertCommonType(t, typedefInfo, t.Name))
          continue;
        typedefInfo.SwiftWrapper = t.SwiftType;

        Writer->addTypedef(t.Name, typedefInfo, swiftVersion);
      }
    }

    bool convertModule() {
      // Set up the writer.
      // FIXME: This is kindof ugly.
      APINotesWriter writer(TheModule.Name, SourceFile);
      Writer = &writer;

      // Write the top-level items.
      convertTopLevelItems(TheModule.TopLevel, VersionTuple());

      if (TheModule.SwiftInferImportAsMember) {
        ModuleOptions opts;
        opts.SwiftInferImportAsMember = true;
        Writer->addModuleOptions(opts);
      }

      // Convert the versioned information.
      for (const auto &versioned : TheModule.SwiftVersions) {
        convertTopLevelItems(versioned.Items, versioned.Version);
      }

      if (!ErrorOccured)
        Writer->writeToStream(OS);

      return ErrorOccured;
    }
  };
}

static bool compile(const Module &module,
                    const FileEntry *sourceFile,
                    llvm::raw_ostream &os,
                    llvm::SourceMgr::DiagHandlerTy diagHandler,
                    void *diagHandlerCtxt){
  using namespace api_notes;

  YAMLConverter c(module, sourceFile, os, diagHandler, diagHandlerCtxt);
  return c.convertModule();
}

/// Simple diagnostic handler that prints diagnostics to standard error.
static void printDiagnostic(const llvm::SMDiagnostic &diag, void *context) {
  diag.print(nullptr, llvm::errs());
}

bool api_notes::compileAPINotes(StringRef yamlInput,
                                const FileEntry *sourceFile,
                                llvm::raw_ostream &os,
                                llvm::SourceMgr::DiagHandlerTy diagHandler,
                                void *diagHandlerCtxt) {
  Module module;

  if (!diagHandler) {
    diagHandler = &printDiagnostic;
  }

  if (parseAPINotes(yamlInput, module, diagHandler, diagHandlerCtxt))
    return true;

  return compile(module, sourceFile, os, diagHandler, diagHandlerCtxt);
}
