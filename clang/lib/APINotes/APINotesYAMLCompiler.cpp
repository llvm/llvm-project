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
    OSType TargetOS;
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
                  OSType targetOS,
                  llvm::raw_ostream &os,
                  llvm::SourceMgr::DiagHandlerTy diagHandler,
                  void *diagHandlerCtxt) :
      TheModule(module), SourceFile(sourceFile), Writer(0), TargetOS(targetOS), OS(os),
      DiagHandler(diagHandler), DiagHandlerCtxt(diagHandlerCtxt),
      ErrorOccured(false) {}

    bool isAvailable(const AvailabilityItem &in) {
      // Check if the API is available on the OS for which we are building.
      if (in.Mode == APIAvailability::OSX && TargetOS != OSType::OSX)
        return false;
      if (in.Mode == APIAvailability::IOS && TargetOS != OSType::IOS)
        return false;
      return true;
    }

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
      if (!isAvailable(common.Availability))
        return true;

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
        if (!isAvailable(prop.Availability))
          continue;
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
        if (!isAvailable(global.Availability))
          continue;
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
        if (!isAvailable(function.Availability))
          continue;
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
        if (!isAvailable(enumConstant.Availability))
          continue;
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
      if (!isAvailable(TheModule.Availability))
        return false;

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
                    api_notes::OSType targetOS,
                    llvm::SourceMgr::DiagHandlerTy diagHandler,
                    void *diagHandlerCtxt){
  using namespace api_notes;

  YAMLConverter c(module, sourceFile, targetOS, os, diagHandler, diagHandlerCtxt);
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
                                OSType targetOS,
                                llvm::SourceMgr::DiagHandlerTy diagHandler,
                                void *diagHandlerCtxt) {
  Module module;

  if (!diagHandler) {
    diagHandler = &printDiagnostic;
  }

  if (parseAPINotes(yamlInput, module, diagHandler, diagHandlerCtxt))
    return true;

  return compile(module, sourceFile, os, targetOS, diagHandler, diagHandlerCtxt);
}

namespace {
  // Deserialize the API notes file into a module.
  class DecompileVisitor : public APINotesReader::Visitor {
    /// Allocator used to clone those strings that need it.
    llvm::BumpPtrAllocator Allocator;

    /// The module we're building.
    Module TheModule;

    /// A known context, which tracks what we know about a context ID.
    struct KnownContext {
      /// Whether this is a protocol (vs. a class).
      bool isProtocol;

      /// The indices into the top-level items for this context at each
      /// Swift version.
      SmallVector<std::pair<VersionTuple, unsigned>, 1> indices;

      Class &getContext(const VersionTuple &swiftVersion,
                        TopLevelItems &items) {
        ClassesSeq &seq = isProtocol ? items.Protocols : items.Classes;

        for (auto &index : indices) {
          if (index.first == swiftVersion)
            return seq[index.second];
        }

        indices.push_back({swiftVersion, seq.size()});
        seq.push_back(Class());
        return seq.back();
      }
    };

    /// A mapping from context ID to a pair (index, is-protocol) that indicates
    /// the index of that class or protocol in the global "classes" or
    /// "protocols" list.
    llvm::DenseMap<unsigned, KnownContext> knownContexts;

    /// Copy a string into allocated memory so it does disappear on us.
    StringRef copyString(StringRef string) {
      if (string.empty()) return StringRef();

      void *ptr = Allocator.Allocate(string.size(), 1);
      memcpy(ptr, string.data(), string.size());
      return StringRef(reinterpret_cast<const char *>(ptr), string.size());
    }

    /// Copy an optional string into allocated memory so it does disappear on us.
    Optional<StringRef> maybeCopyString(Optional<StringRef> string) {
      if (!string) return None;

      return copyString(*string);
    }

    /// Copy an optional string into allocated memory so it does disappear on us.
    Optional<StringRef> maybeCopyString(Optional<std::string> string) {
      if (!string) return None;

      return copyString(*string);
    }

    template<typename T>
    void handleCommon(T &record, const CommonEntityInfo &info) {
      handleAvailability(record.Availability, info);
      record.SwiftPrivate = info.isSwiftPrivate();
      record.SwiftName = copyString(info.SwiftName);
    }

    template<typename T>
    void handleCommonType(T &record, const CommonTypeInfo &info) {
      handleCommon(record, info);
      record.SwiftBridge = maybeCopyString(info.getSwiftBridge());
      record.NSErrorDomain = maybeCopyString(info.getNSErrorDomain());
    }

    /// Map Objective-C context info.
    void handleObjCContext(Class &record, StringRef name,
                           const ObjCContextInfo &info) {
      record.Name = name;

      handleCommonType(record, info);
      record.SwiftImportAsNonGeneric = info.getSwiftImportAsNonGeneric();
      record.SwiftObjCMembers = info.getSwiftObjCMembers();

      if (info.getDefaultNullability()) {
        record.AuditedForNullability = true;
      }
    }

    /// Map availability information, if present.
    void handleAvailability(AvailabilityItem &availability,
                            const CommonEntityInfo &info) {
      if (info.Unavailable) {
        availability.Mode = APIAvailability::None;
        availability.Msg = copyString(info.UnavailableMsg);
      }

      if (info.UnavailableInSwift) {
        availability.Mode = APIAvailability::NonSwift;
        availability.Msg = copyString(info.UnavailableMsg);
      }
    }

    /// Map parameter information for a function.
    void handleParameters(ParamsSeq &params,
                          const FunctionInfo &info) {
      unsigned position = 0;
      for (const auto &pi: info.Params) {
        Param p;
        p.Position = position++;
        p.Nullability = pi.getNullability();
        p.NoEscape = pi.isNoEscape();
        p.Type = copyString(pi.getType());
        p.RetainCountConvention = pi.getRetainCountConvention();
        params.push_back(p);
      }
    }

    /// Map nullability information for a function.
    void handleNullability(NullabilitySeq &nullability,
                           llvm::Optional<NullabilityKind> &nullabilityOfRet,
                           const FunctionInfo &info,
                           unsigned numParams) {
      if (info.NullabilityAudited) {
        nullabilityOfRet = info.getReturnTypeInfo();

        // Figure out the number of parameters from the selector.
        for (unsigned i = 0; i != numParams; ++i)
          nullability.push_back(info.getParamTypeInfo(i));
      }
    }

    TopLevelItems &getTopLevelItems(VersionTuple swiftVersion) {
      if (!swiftVersion) return TheModule.TopLevel;

      for (auto &versioned : TheModule.SwiftVersions) {
        if (versioned.Version == swiftVersion)
          return versioned.Items;
      }

      TheModule.SwiftVersions.push_back(Versioned());
      TheModule.SwiftVersions.back().Version = swiftVersion;
      return TheModule.SwiftVersions.back().Items;
    }

  public:
    virtual void visitObjCClass(ContextID contextID, StringRef name,
                                const ObjCContextInfo &info,
                                VersionTuple swiftVersion) {
      // Record this known context.
      auto &items = getTopLevelItems(swiftVersion);
      auto &known = knownContexts[contextID.Value];
      known.isProtocol = false;

      handleObjCContext(known.getContext(swiftVersion, items), name, info);
    }

    virtual void visitObjCProtocol(ContextID contextID, StringRef name,
                                   const ObjCContextInfo &info,
                                   VersionTuple swiftVersion) {
      // Record this known context.
      auto &items = getTopLevelItems(swiftVersion);
      auto &known = knownContexts[contextID.Value];
      known.isProtocol = true;

      handleObjCContext(known.getContext(swiftVersion, items), name, info);
    }

    virtual void visitObjCMethod(ContextID contextID, StringRef selector,
                                 bool isInstanceMethod,
                                 const ObjCMethodInfo &info,
                                 VersionTuple swiftVersion) {
      Method method;
      method.Selector = copyString(selector);
      method.Kind = isInstanceMethod ? MethodKind::Instance : MethodKind::Class;

      handleCommon(method, info);
      handleParameters(method.Params, info);
      handleNullability(method.Nullability, method.NullabilityOfRet, info,
                        selector.count(':'));
      method.DesignatedInit = info.DesignatedInit;
      method.Required = info.Required;
      method.ResultType = copyString(info.ResultType);
      method.RetainCountConvention = info.getRetainCountConvention();
      auto &items = getTopLevelItems(swiftVersion);
      knownContexts[contextID.Value].getContext(swiftVersion, items)
        .Methods.push_back(method);
    }

    virtual void visitObjCProperty(ContextID contextID, StringRef name,
                                   bool isInstance,
                                   const ObjCPropertyInfo &info,
                                   VersionTuple swiftVersion) {
      Property property;
      property.Name = name;
      property.Kind = isInstance ? MethodKind::Instance : MethodKind::Class;
      handleCommon(property, info);

      // FIXME: No way to represent "not audited for nullability".
      if (auto nullability = info.getNullability()) {
        property.Nullability = *nullability;
      }

      property.SwiftImportAsAccessors = info.getSwiftImportAsAccessors();

      property.Type = copyString(info.getType());

      auto &items = getTopLevelItems(swiftVersion);
      knownContexts[contextID.Value].getContext(swiftVersion, items)
        .Properties.push_back(property);
    }

    virtual void visitGlobalFunction(StringRef name,
                                     const GlobalFunctionInfo &info,
                                     VersionTuple swiftVersion) {
      Function function;
      function.Name = name;
      handleCommon(function, info);
      handleParameters(function.Params, info);
      if (info.NumAdjustedNullable > 0)
        handleNullability(function.Nullability, function.NullabilityOfRet,
                          info, info.NumAdjustedNullable-1);
      function.ResultType = copyString(info.ResultType);
      function.RetainCountConvention = info.getRetainCountConvention();
      auto &items = getTopLevelItems(swiftVersion);
      items.Functions.push_back(function);
    }

    virtual void visitGlobalVariable(StringRef name,
                                     const GlobalVariableInfo &info,
                                     VersionTuple swiftVersion) {
      GlobalVariable global;
      global.Name = name;
      handleCommon(global, info);

      // FIXME: No way to represent "not audited for nullability".
      if (auto nullability = info.getNullability()) {
        global.Nullability = *nullability;
      }
      global.Type = copyString(info.getType());

      auto &items = getTopLevelItems(swiftVersion);
      items.Globals.push_back(global);
    }

    virtual void visitEnumConstant(StringRef name,
                                   const EnumConstantInfo &info,
                                   VersionTuple swiftVersion) {
      EnumConstant enumConstant;
      enumConstant.Name = name;
      handleCommon(enumConstant, info);

      auto &items = getTopLevelItems(swiftVersion);
      items.EnumConstants.push_back(enumConstant);
    }

    virtual void visitTag(StringRef name, const TagInfo &info,
                          VersionTuple swiftVersion) {
      Tag tag;
      tag.Name = name;
      handleCommonType(tag, info);
      tag.EnumExtensibility = info.EnumExtensibility;
      tag.FlagEnum = info.isFlagEnum();
      auto &items = getTopLevelItems(swiftVersion);
      items.Tags.push_back(tag);
    }

    virtual void visitTypedef(StringRef name, const TypedefInfo &info,
                              VersionTuple swiftVersion) {
      Typedef td;
      td.Name = name;
      handleCommonType(td, info);
      td.SwiftWrapper = info.SwiftWrapper;
      auto &items = getTopLevelItems(swiftVersion);
      items.Typedefs.push_back(td);
    }

    /// Retrieve the module.
    Module &getModule() { return TheModule; }
  };
}

/// Produce a flattened, numeric value for optional method/property kinds.
static unsigned flattenPropertyKind(llvm::Optional<MethodKind> kind) {
  return kind ? (*kind == MethodKind::Instance ? 2 : 1) : 0;
}

/// Sort the items in the given block of "top-level" items.
static void sortTopLevelItems(TopLevelItems &items) {
  // Sort classes.
  std::sort(items.Classes.begin(), items.Classes.end(),
            [](const Class &lhs, const Class &rhs) -> bool {
              return lhs.Name < rhs.Name;
            });

  // Sort protocols.
  std::sort(items.Protocols.begin(), items.Protocols.end(),
            [](const Class &lhs, const Class &rhs) -> bool {
              return lhs.Name < rhs.Name;
            });

  // Sort methods and properties within each class and protocol.
  auto sortMembers = [](Class &record) {
    // Sort properties.
    std::sort(record.Properties.begin(), record.Properties.end(),
              [](const Property &lhs, const Property &rhs) -> bool {
                return lhs.Name < rhs.Name ||
                (lhs.Name == rhs.Name &&
                 flattenPropertyKind(lhs.Kind) <
                 flattenPropertyKind(rhs.Kind));
              });

    // Sort methods.
    std::sort(record.Methods.begin(), record.Methods.end(),
              [](const Method &lhs, const Method &rhs) -> bool {
                return lhs.Selector < rhs.Selector ||
                (lhs.Selector == rhs.Selector &&
                 static_cast<unsigned>(lhs.Kind)
                 < static_cast<unsigned>(rhs.Kind));
              });
  };
  std::for_each(items.Classes.begin(), items.Classes.end(), sortMembers);
  std::for_each(items.Protocols.begin(), items.Protocols.end(), sortMembers);

  // Sort functions.
  std::sort(items.Functions.begin(), items.Functions.end(),
            [](const Function &lhs, const Function &rhs) -> bool {
              return lhs.Name < rhs.Name;
            });

  // Sort global variables.
  std::sort(items.Globals.begin(), items.Globals.end(),
            [](const GlobalVariable &lhs, const GlobalVariable &rhs) -> bool {
              return lhs.Name < rhs.Name;
            });

  // Sort enum constants.
  std::sort(items.EnumConstants.begin(), items.EnumConstants.end(),
            [](const EnumConstant &lhs, const EnumConstant &rhs) -> bool {
              return lhs.Name < rhs.Name;
            });

  // Sort tags.
  std::sort(items.Tags.begin(), items.Tags.end(),
            [](const Tag &lhs, const Tag &rhs) -> bool {
              return lhs.Name < rhs.Name;
            });

  // Sort typedefs.
  std::sort(items.Typedefs.begin(), items.Typedefs.end(),
            [](const Typedef &lhs, const Typedef &rhs) -> bool {
              return lhs.Name < rhs.Name;
            });
}

bool api_notes::decompileAPINotes(std::unique_ptr<llvm::MemoryBuffer> input,
                                  llvm::raw_ostream &os) {
  // Try to read the file.
  auto reader = APINotesReader::get(std::move(input), VersionTuple());
  if (!reader) {
    llvm::errs() << "not a well-formed API notes binary file\n";
    return true;
  }

  DecompileVisitor decompileVisitor;
  reader->visit(decompileVisitor);

  // Sort the data in the module, because the API notes reader doesn't preserve
  // order.
  auto &module = decompileVisitor.getModule();

  // Set module name.
  module.Name = reader->getModuleName();

  // Set module options
  auto opts = reader->getModuleOptions();
  if (opts.SwiftInferImportAsMember)
    module.SwiftInferImportAsMember = true;

  // Sort the top-level items.
  sortTopLevelItems(module.TopLevel);

  // Sort the Swift versions.
  std::sort(module.SwiftVersions.begin(), module.SwiftVersions.end(),
            [](const Versioned &lhs, const Versioned &rhs) -> bool {
              return lhs.Version < rhs.Version;
            });

  // Sort the top-level items within each Swift version.
  for (auto &versioned : module.SwiftVersions)
    sortTopLevelItems(versioned.Items);

  // Output the YAML representation.
  Output yout(os);
  yout << module;

  return false;
}

