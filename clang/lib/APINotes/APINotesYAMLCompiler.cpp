//===--- APINotesYAMLCompiler.cpp - API Notes YAML format reader *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file reads API notes specified in YAML format.
//
//===----------------------------------------------------------------------===//
#include "clang/APINotes/APINotesYAMLCompiler.h"
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

using llvm::StringRef;
using namespace clang;
namespace {
  enum class APIAvailability {
    Available = 0,
    OSX,
    IOS,
    None,
    NonSwift,
  };

  enum class MethodKind {
    Class,
    Instance,
  };

  /// Old attribute deprecated in favor of SwiftName.
  enum class FactoryAsInitKind {
    /// Infer based on name and type (the default).
    Infer,
    /// Treat as a class method.
    AsClassMethod,
    /// Treat as an initializer.
    AsInitializer
  };
  
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

  struct AvailabilityItem {
    APIAvailability Mode = APIAvailability::Available;
    StringRef Msg;
    AvailabilityItem() : Mode(APIAvailability::Available), Msg("") {}
  };

  static llvm::Optional<NullabilityKind> AbsentNullability = llvm::None;
  static llvm::Optional<NullabilityKind> DefaultNullability =
    NullabilityKind::NonNull;
  typedef std::vector<clang::NullabilityKind> NullabilitySeq;

  struct Param {
    unsigned Position;
    Optional<bool> NoEscape = false;
    Optional<NullabilityKind> Nullability;
    Optional<api_notes::RetainCountConventionKind> RetainCountConvention;
    StringRef Type;
  };
  typedef std::vector<Param> ParamsSeq;

  struct Method {
    StringRef Selector;
    MethodKind Kind;
    ParamsSeq Params;
    NullabilitySeq Nullability;
    Optional<NullabilityKind> NullabilityOfRet;
    Optional<api_notes::RetainCountConventionKind> RetainCountConvention;
    AvailabilityItem Availability;
    Optional<bool> SwiftPrivate;
    StringRef SwiftName;
    FactoryAsInitKind FactoryAsInit = FactoryAsInitKind::Infer;
    bool DesignatedInit = false;
    bool Required = false;
    StringRef ResultType;
  };
  typedef std::vector<Method> MethodsSeq;

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

  struct GlobalVariable {
    StringRef Name;
    llvm::Optional<NullabilityKind> Nullability;
    AvailabilityItem Availability;
    Optional<bool> SwiftPrivate;
    StringRef SwiftName;
    StringRef Type;
  };
  typedef std::vector<GlobalVariable> GlobalVariablesSeq;

  struct EnumConstant {
    StringRef Name;
    AvailabilityItem Availability;
    Optional<bool> SwiftPrivate;
    StringRef SwiftName;
  };
  typedef std::vector<EnumConstant> EnumConstantsSeq;

  struct Tag {
    StringRef Name;
    AvailabilityItem Availability;
    StringRef SwiftName;
    Optional<bool> SwiftPrivate;
    Optional<StringRef> SwiftBridge;
    Optional<StringRef> NSErrorDomain;
    Optional<api_notes::EnumExtensibilityKind> EnumExtensibility;
    Optional<bool> FlagEnum;
    Optional<EnumConvenienceAliasKind> EnumConvenienceKind;
  };
  typedef std::vector<Tag> TagsSeq;

  struct Typedef {
    StringRef Name;
    AvailabilityItem Availability;
    StringRef SwiftName;
    Optional<bool> SwiftPrivate;
    Optional<StringRef> SwiftBridge;
    Optional<StringRef> NSErrorDomain;
    Optional<api_notes::SwiftWrapperKind> SwiftWrapper;
  };
  typedef std::vector<Typedef> TypedefsSeq;

  struct TopLevelItems {
    ClassesSeq Classes;
    ClassesSeq Protocols;
    FunctionsSeq Functions;
    GlobalVariablesSeq Globals;
    EnumConstantsSeq EnumConstants;
    TagsSeq Tags;
    TypedefsSeq Typedefs;
  };

  struct Versioned {
    VersionTuple Version;
    TopLevelItems Items;
  };

  typedef std::vector<Versioned> VersionedSeq;

  struct Module {
    StringRef Name;
    AvailabilityItem Availability;
    TopLevelItems TopLevel;
    VersionedSeq SwiftVersions;

    llvm::Optional<bool> SwiftInferImportAsMember = {llvm::None};

    LLVM_ATTRIBUTE_DEPRECATED(
      void dump() LLVM_ATTRIBUTE_USED,
      "only for use within the debugger");
  };
}

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(clang::NullabilityKind)
LLVM_YAML_IS_SEQUENCE_VECTOR(Method)
LLVM_YAML_IS_SEQUENCE_VECTOR(Property)
LLVM_YAML_IS_SEQUENCE_VECTOR(Param)
LLVM_YAML_IS_SEQUENCE_VECTOR(Class)
LLVM_YAML_IS_SEQUENCE_VECTOR(Function)
LLVM_YAML_IS_SEQUENCE_VECTOR(GlobalVariable)
LLVM_YAML_IS_SEQUENCE_VECTOR(EnumConstant)
LLVM_YAML_IS_SEQUENCE_VECTOR(Tag)
LLVM_YAML_IS_SEQUENCE_VECTOR(Typedef)
LLVM_YAML_IS_SEQUENCE_VECTOR(Versioned)

namespace llvm {
  namespace yaml {

    template <>
    struct ScalarEnumerationTraits<NullabilityKind > {
      static void enumeration(IO &io, NullabilityKind  &value) {
        io.enumCase(value, "N", NullabilityKind::NonNull);
        io.enumCase(value, "O", NullabilityKind::Nullable);
        io.enumCase(value, "U", NullabilityKind::Unspecified);
        // TODO: Mapping this to it's own value would allow for better cross
        // checking. Also the default should be Unknown.
        io.enumCase(value, "S", NullabilityKind::Unspecified);
      }
    };

    template <>
    struct ScalarEnumerationTraits<FactoryAsInitKind> {
      static void enumeration(IO &io, FactoryAsInitKind  &value) {
        io.enumCase(value, "A", FactoryAsInitKind::Infer);
        io.enumCase(value, "C", FactoryAsInitKind::AsClassMethod);
        io.enumCase(value, "I", FactoryAsInitKind::AsInitializer);
      }
    };

    template <>
    struct ScalarEnumerationTraits<MethodKind> {
      static void enumeration(IO &io, MethodKind &value) {
        io.enumCase(value, "Class",    MethodKind::Class);
        io.enumCase(value, "Instance", MethodKind::Instance);
      }
    };

    template <>
    struct ScalarEnumerationTraits<APIAvailability> {
      static void enumeration(IO &io, APIAvailability &value) {
        io.enumCase(value, "OSX",       APIAvailability::OSX);
        io.enumCase(value, "iOS",       APIAvailability::IOS);
        io.enumCase(value, "none",      APIAvailability::None);
        io.enumCase(value, "nonswift",  APIAvailability::NonSwift);
        io.enumCase(value, "available", APIAvailability::Available);
      }
    };

    template<>
    struct ScalarEnumerationTraits<api_notes::SwiftWrapperKind> {
      static void enumeration(IO &io, api_notes::SwiftWrapperKind &value) {
        io.enumCase(value, "none",      api_notes::SwiftWrapperKind::None);
        io.enumCase(value, "struct",    api_notes::SwiftWrapperKind::Struct);
        io.enumCase(value, "enum",      api_notes::SwiftWrapperKind::Enum);
      }
    };

    template<>
    struct ScalarEnumerationTraits<api_notes::EnumExtensibilityKind> {
      static void enumeration(IO &io, api_notes::EnumExtensibilityKind &value) {
        io.enumCase(value, "none",   api_notes::EnumExtensibilityKind::None);
        io.enumCase(value, "open",   api_notes::EnumExtensibilityKind::Open);
        io.enumCase(value, "closed", api_notes::EnumExtensibilityKind::Closed);
      }
    };

    template<>
    struct ScalarEnumerationTraits<EnumConvenienceAliasKind> {
      static void enumeration(IO &io, EnumConvenienceAliasKind &value) {
        io.enumCase(value, "none",      EnumConvenienceAliasKind::None);
        io.enumCase(value, "CFEnum",    EnumConvenienceAliasKind::CFEnum);
        io.enumCase(value, "NSEnum",    EnumConvenienceAliasKind::CFEnum);
        io.enumCase(value, "CFOptions", EnumConvenienceAliasKind::CFOptions);
        io.enumCase(value, "NSOptions", EnumConvenienceAliasKind::CFOptions);
        io.enumCase(value, "CFClosedEnum",
                    EnumConvenienceAliasKind::CFClosedEnum);
        io.enumCase(value, "NSClosedEnum",
                    EnumConvenienceAliasKind::CFClosedEnum);
      }
    };

    template<>
    struct ScalarEnumerationTraits<api_notes::RetainCountConventionKind> {
      static void enumeration(IO &io,
                              api_notes::RetainCountConventionKind &value) {
        using api_notes::RetainCountConventionKind;
        io.enumCase(value, "none", RetainCountConventionKind::None);
        io.enumCase(value, "CFReturnsRetained",
                    RetainCountConventionKind::CFReturnsRetained);
        io.enumCase(value, "CFReturnsNotRetained",
                    RetainCountConventionKind::CFReturnsNotRetained);
        io.enumCase(value, "NSReturnsRetained",
                    RetainCountConventionKind::NSReturnsRetained);
        io.enumCase(value, "NSReturnsNotRetained",
                    RetainCountConventionKind::NSReturnsNotRetained);
      }
    };

    template <>
    struct ScalarTraits<VersionTuple> {
      static void output(const VersionTuple &value, void*,
                         llvm::raw_ostream &out) {
        out << value;
      }
      static StringRef input(StringRef scalar, void*, VersionTuple &value) {
        if (value.tryParse(scalar))
          return "not a version number in the form XX.YY";

        return StringRef();
      }

      static QuotingType mustQuote(StringRef) { return QuotingType::None; }
    };

    template <>
    struct MappingTraits<Param> {
      static void mapping(IO &io, Param& p) {
        io.mapRequired("Position",              p.Position);
        io.mapOptional("Nullability",           p.Nullability, 
                                                AbsentNullability);
        io.mapOptional("RetainCountConvention", p.RetainCountConvention);
        io.mapOptional("NoEscape",              p.NoEscape);
        io.mapOptional("Type",                  p.Type, StringRef(""));
      }
    };

    template <>
    struct MappingTraits<Property> {
      static void mapping(IO &io, Property& p) {
        io.mapRequired("Name",            p.Name);
        io.mapOptional("PropertyKind",    p.Kind);
        io.mapOptional("Nullability",     p.Nullability, 
                                          AbsentNullability);
        io.mapOptional("Availability",    p.Availability.Mode);
        io.mapOptional("AvailabilityMsg", p.Availability.Msg);
        io.mapOptional("SwiftPrivate",    p.SwiftPrivate);
        io.mapOptional("SwiftName",       p.SwiftName);
        io.mapOptional("SwiftImportAsAccessors", p.SwiftImportAsAccessors);
        io.mapOptional("Type",            p.Type, StringRef(""));
      }
    };

    template <>
    struct MappingTraits<Method> {
      static void mapping(IO &io, Method& m) {
        io.mapRequired("Selector",              m.Selector);
        io.mapRequired("MethodKind",            m.Kind);
        io.mapOptional("Parameters",            m.Params);
        io.mapOptional("Nullability",           m.Nullability);
        io.mapOptional("NullabilityOfRet",      m.NullabilityOfRet,
                                                AbsentNullability);
        io.mapOptional("RetainCountConvention", m.RetainCountConvention);
        io.mapOptional("Availability",          m.Availability.Mode);
        io.mapOptional("AvailabilityMsg",       m.Availability.Msg);
        io.mapOptional("SwiftPrivate",          m.SwiftPrivate);
        io.mapOptional("SwiftName",             m.SwiftName);
        io.mapOptional("FactoryAsInit",         m.FactoryAsInit,
                                                FactoryAsInitKind::Infer);
        io.mapOptional("DesignatedInit",        m.DesignatedInit, false);
        io.mapOptional("Required",              m.Required, false);
        io.mapOptional("ResultType",            m.ResultType, StringRef(""));
      }
    };

    template <>
    struct MappingTraits<Class> {
      static void mapping(IO &io, Class& c) {
        io.mapRequired("Name",                  c.Name);
        io.mapOptional("AuditedForNullability", c.AuditedForNullability, false);
        io.mapOptional("Availability",          c.Availability.Mode);
        io.mapOptional("AvailabilityMsg",       c.Availability.Msg);
        io.mapOptional("SwiftPrivate",          c.SwiftPrivate);
        io.mapOptional("SwiftName",             c.SwiftName);
        io.mapOptional("SwiftBridge",           c.SwiftBridge);
        io.mapOptional("NSErrorDomain",         c.NSErrorDomain);
        io.mapOptional("SwiftImportAsNonGeneric", c.SwiftImportAsNonGeneric);
        io.mapOptional("SwiftObjCMembers",      c.SwiftObjCMembers);
        io.mapOptional("Methods",               c.Methods);
        io.mapOptional("Properties",            c.Properties);
      }
    };

    template <>
    struct MappingTraits<Function> {
      static void mapping(IO &io, Function& f) {
        io.mapRequired("Name",                  f.Name);
        io.mapOptional("Parameters",            f.Params);
        io.mapOptional("Nullability",           f.Nullability);
        io.mapOptional("NullabilityOfRet",      f.NullabilityOfRet,
                                                AbsentNullability);
        io.mapOptional("RetainCountConvention", f.RetainCountConvention);
        io.mapOptional("Availability",          f.Availability.Mode);
        io.mapOptional("AvailabilityMsg",       f.Availability.Msg);
        io.mapOptional("SwiftPrivate",          f.SwiftPrivate);
        io.mapOptional("SwiftName",             f.SwiftName);
        io.mapOptional("ResultType",            f.ResultType, StringRef(""));
      }
    };

    template <>
    struct MappingTraits<GlobalVariable> {
      static void mapping(IO &io, GlobalVariable& v) {
        io.mapRequired("Name",            v.Name);
        io.mapOptional("Nullability",     v.Nullability,
                                          AbsentNullability);
        io.mapOptional("Availability",    v.Availability.Mode);
        io.mapOptional("AvailabilityMsg", v.Availability.Msg);
        io.mapOptional("SwiftPrivate",    v.SwiftPrivate);
        io.mapOptional("SwiftName",       v.SwiftName);
        io.mapOptional("Type",            v.Type, StringRef(""));
      }
    };

    template <>
    struct MappingTraits<EnumConstant> {
      static void mapping(IO &io, EnumConstant& v) {
        io.mapRequired("Name",            v.Name);
        io.mapOptional("Availability",    v.Availability.Mode);
        io.mapOptional("AvailabilityMsg", v.Availability.Msg);
        io.mapOptional("SwiftPrivate",    v.SwiftPrivate);
        io.mapOptional("SwiftName",       v.SwiftName);
      }
    };

    template <>
    struct MappingTraits<Tag> {
      static void mapping(IO &io, Tag& t) {
        io.mapRequired("Name",                  t.Name);
        io.mapOptional("Availability",          t.Availability.Mode);
        io.mapOptional("AvailabilityMsg",       t.Availability.Msg);
        io.mapOptional("SwiftPrivate",          t.SwiftPrivate);
        io.mapOptional("SwiftName",             t.SwiftName);
        io.mapOptional("SwiftBridge",           t.SwiftBridge);
        io.mapOptional("NSErrorDomain",         t.NSErrorDomain);
        io.mapOptional("EnumExtensibility",     t.EnumExtensibility);
        io.mapOptional("FlagEnum",              t.FlagEnum);
        io.mapOptional("EnumKind",              t.EnumConvenienceKind);
      }
    };

    template <>
    struct MappingTraits<Typedef> {
      static void mapping(IO &io, Typedef& t) {
        io.mapRequired("Name",                  t.Name);
        io.mapOptional("Availability",          t.Availability.Mode);
        io.mapOptional("AvailabilityMsg",       t.Availability.Msg);
        io.mapOptional("SwiftPrivate",          t.SwiftPrivate);
        io.mapOptional("SwiftName",             t.SwiftName);
        io.mapOptional("SwiftBridge",           t.SwiftBridge);
        io.mapOptional("NSErrorDomain",         t.NSErrorDomain);
        io.mapOptional("SwiftWrapper",         t.SwiftWrapper);
      }
    };

    static void mapTopLevelItems(IO &io, TopLevelItems &i) {
      io.mapOptional("Classes",         i.Classes);
      io.mapOptional("Protocols",       i.Protocols);
      io.mapOptional("Functions",       i.Functions);
      io.mapOptional("Globals",         i.Globals);
      io.mapOptional("Enumerators",     i.EnumConstants);
      io.mapOptional("Tags",            i.Tags);
      io.mapOptional("Typedefs",        i.Typedefs);
    }

    template <>
    struct MappingTraits<Versioned> {
      static void mapping(IO &io, Versioned& v) {
        io.mapRequired("Version", v.Version);
        mapTopLevelItems(io, v.Items);
      }
    };

    template <>
    struct MappingTraits<Module> {
      static void mapping(IO &io, Module& m) {
        io.mapRequired("Name",            m.Name);
        io.mapOptional("Availability",    m.Availability.Mode);
        io.mapOptional("AvailabilityMsg", m.Availability.Msg);
        io.mapOptional("SwiftInferImportAsMember", m.SwiftInferImportAsMember);

        mapTopLevelItems(io, m.TopLevel);

        io.mapOptional("SwiftVersions",  m.SwiftVersions);
      }
    };
  }
}

using llvm::yaml::Input;
using llvm::yaml::Output;

void Module::dump() {
  Output yout(llvm::errs());
  yout << *this;
}

static bool parseAPINotes(StringRef yamlInput, Module &module,
                          llvm::SourceMgr::DiagHandlerTy diagHandler,
                          void *diagHandlerCtxt) {
  Input yin(yamlInput, nullptr, diagHandler, diagHandlerCtxt);
  yin >> module;

  return static_cast<bool>(yin.error());
}

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
        outInfo.UnavailableMsg = in.Msg;
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
        pi.setType(p.Type);
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
        outInfo.addTypeInfo(0, *DefaultNullability);
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
      info.SwiftName = common.SwiftName;
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
      mInfo.Required = meth.Required;
      if (meth.FactoryAsInit != FactoryAsInitKind::Infer) {
        emitError("'FactoryAsInit' is no longer valid; "
                  "use 'SwiftName' instead");
      }
      mInfo.ResultType = meth.ResultType;

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
        cInfo.setDefaultNullability(*DefaultNullability);
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
        pInfo.SwiftName = prop.SwiftName;
        if (prop.Nullability)
          pInfo.setNullabilityAudited(*prop.Nullability);
        if (prop.SwiftImportAsAccessors)
          pInfo.setSwiftImportAsAccessors(*prop.SwiftImportAsAccessors);
        pInfo.setType(prop.Type);
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
        info.SwiftName = global.SwiftName;
        if (global.Nullability)
          info.setNullabilityAudited(*global.Nullability);
        info.setType(global.Type);
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
        info.SwiftName = function.SwiftName;
        convertParams(function.Params, info);
        convertNullability(function.Nullability,
                           function.NullabilityOfRet,
                           info, function.Name);
        info.ResultType = function.ResultType;
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
        info.SwiftName = enumConstant.SwiftName;
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
        typedefInfo.SwiftWrapper = t.SwiftWrapper;

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

bool api_notes::parseAndDumpAPINotes(StringRef yamlInput)  {
  Module module;

  if (parseAPINotes(yamlInput, module, nullptr, nullptr))
    return true;

  Output yout(llvm::outs());
  yout << module;

  return false;
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
