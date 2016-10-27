//===--- LangOptions.h - C Language Family Language Options -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::LangOptions interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_LANGOPTIONS_H
#define LLVM_CLANG_BASIC_LANGOPTIONS_H

#include "clang/Basic/CommentOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/ObjCRuntime.h"
#include "clang/Basic/Sanitizers.h"
#include "clang/Basic/Visibility.h"
#include <string>
#include <vector>

namespace clang {

/// Bitfields of LangOptions, split out from LangOptions in order to ensure that
/// this large collection of bitfields is a trivial class type.
class LangOptionsBase {
public:
  // Define simple language options (with no accessors).
#define LANGOPT(Name, Bits, Default, Description) unsigned Name : Bits;
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description)
#include "clang/Basic/LangOptions.def"

protected:
  // Define language options of enumeration type. These are private, and will
  // have accessors (below).
#define LANGOPT(Name, Bits, Default, Description)
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  unsigned Name : Bits;
#include "clang/Basic/LangOptions.def"
};

/// \brief Keeps track of the various options that can be
/// enabled, which controls the dialect of C or C++ that is accepted.
class LangOptions : public LangOptionsBase {
public:
  typedef clang::Visibility Visibility;
  
  enum GCMode { NonGC, GCOnly, HybridGC };
  enum StackProtectorMode { SSPOff, SSPOn, SSPStrong, SSPReq };
  
  enum SignedOverflowBehaviorTy {
    SOB_Undefined,  // Default C standard behavior.
    SOB_Defined,    // -fwrapv
    SOB_Trapping    // -ftrapv
  };

  enum CompilingModuleKind {
    CMK_None,           ///< Not compiling a module interface at all.
    CMK_ModuleMap,      ///< Compiling a module from a module map.
    CMK_ModuleInterface ///< Compiling a C++ modules TS module interface unit.
  };

  enum PragmaMSPointersToMembersKind {
    PPTMK_BestCase,
    PPTMK_FullGeneralitySingleInheritance,
    PPTMK_FullGeneralityMultipleInheritance,
    PPTMK_FullGeneralityVirtualInheritance
  };

  enum DefaultCallingConvention {
    DCC_None,
    DCC_CDecl,
    DCC_FastCall,
    DCC_StdCall,
    DCC_VectorCall
  };

  enum AddrSpaceMapMangling { ASMM_Target, ASMM_On, ASMM_Off };

  enum MSVCMajorVersion {
    MSVC2010 = 16,
    MSVC2012 = 17,
    MSVC2013 = 18,
    MSVC2015 = 19
  };

public:
  /// \brief Set of enabled sanitizers.
  SanitizerSet Sanitize;

  /// \brief Paths to blacklist files specifying which objects
  /// (files, functions, variables) should not be instrumented.
  std::vector<std::string> SanitizerBlacklistFiles;

  clang::ObjCRuntime ObjCRuntime;

  std::string ObjCConstantStringClass;
  
  /// \brief The name of the handler function to be called when -ftrapv is
  /// specified.
  ///
  /// If none is specified, abort (GCC-compatible behaviour).
  std::string OverflowHandler;

  /// \brief The name of the current module, of which the main source file
  /// is a part. If CompilingModule is set, we are compiling the interface
  /// of this module, otherwise we are compiling an implementation file of
  /// it.
  std::string CurrentModule;

  /// \brief The names of any features to enable in module 'requires' decls
  /// in addition to the hard-coded list in Module.cpp and the target features.
  ///
  /// This list is sorted.
  std::vector<std::string> ModuleFeatures;

  /// \brief Options for parsing comments.
  CommentOptions CommentOpts;

  /// \brief A list of all -fno-builtin-* function names (e.g., memset).
  std::vector<std::string> NoBuiltinFuncs;

  /// \brief Triples of the OpenMP targets that the host code codegen should
  /// take into account in order to generate accurate offloading descriptors.
  std::vector<llvm::Triple> OMPTargetTriples;

  /// \brief Name of the IR file that contains the result of the OpenMP target
  /// host code generation.
  std::string OMPHostIRFile;

  /// \brief Indicates whether the front-end is explicitly told that the
  /// input is a header file (i.e. -x c-header).
  bool IsHeaderFile;

  LangOptions();

  // Define accessors/mutators for language options of enumeration type.
#define LANGOPT(Name, Bits, Default, Description) 
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  Type get##Name() const { return static_cast<Type>(Name); } \
  void set##Name(Type Value) { Name = static_cast<unsigned>(Value); }  
#include "clang/Basic/LangOptions.def"

  /// Are we compiling a module interface (.cppm or module map)?
  bool isCompilingModule() const {
    return getCompilingModule() != CMK_None;
  }

  bool isSignedOverflowDefined() const {
    return getSignedOverflowBehavior() == SOB_Defined;
  }
  
  bool isSubscriptPointerArithmetic() const {
    return ObjCRuntime.isSubscriptPointerArithmetic() &&
           !ObjCSubscriptingLegacyRuntime;
  }

  bool isCompatibleWithMSVC(MSVCMajorVersion MajorVersion) const {
    return MSCompatibilityVersion >= MajorVersion * 10000000U;
  }

  /// \brief Reset all of the options that are not considered when building a
  /// module.
  void resetNonModularOptions();

  /// \brief Is this a libc/libm function that is no longer recognized as a
  /// builtin because a -fno-builtin-* option has been specified?
  bool isNoBuiltinFunc(StringRef Name) const;
};

/// \brief Floating point control options
class FPOptions {
public:
  unsigned fp_contract : 1;

  FPOptions() : fp_contract(0) {}

  FPOptions(const LangOptions &LangOpts) :
    fp_contract(LangOpts.DefaultFPContract) {}
};

/// \brief Describes the kind of translation unit being processed.
enum TranslationUnitKind {
  /// \brief The translation unit is a complete translation unit.
  TU_Complete,
  /// \brief The translation unit is a prefix to a translation unit, and is
  /// not complete.
  TU_Prefix,
  /// \brief The translation unit is a module.
  TU_Module
};
  
}  // end namespace clang

#endif
