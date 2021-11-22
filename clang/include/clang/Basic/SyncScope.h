//===--- SyncScope.h - Atomic synchronization scopes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides definitions for the atomic synchronization scopes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SYNCSCOPE_H
#define LLVM_CLANG_BASIC_SYNCSCOPE_H

#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace clang {

/// Defines synch scope values used internally by clang.
///
/// The enum values start from 0 and are contiguous. They are mainly used for
/// enumerating all supported synch scope values and mapping them to LLVM
/// synch scopes. Their numerical values may be different from the corresponding
/// synch scope enums used in source languages.
///
/// In atomic builtin and expressions, language-specific synch scope enums are
/// used. Currently only OpenCL memory scope enums are supported and assumed
/// to be used by all languages. However, in the future, other languages may
/// define their own set of synch scope enums. The language-specific synch scope
/// values are represented by class AtomicScopeModel and its derived classes.
///
/// To add a new enum value:
///   Add the enum value to enum class SyncScope.
///   Update enum value Last if necessary.
///   Update getAsString.
///
enum class SyncScope {
  HIPSingleThread,
  HIPWavefront,
  HIPWorkgroup,
  HIPAgent,
  HIPSystem,
  OpenCLWorkGroup,
  OpenCLDevice,
  OpenCLAllSVMDevices,
  OpenCLSubGroup,
  Last = OpenCLSubGroup
};

inline llvm::StringRef getAsString(SyncScope S) {
  switch (S) {
  case SyncScope::HIPSingleThread:
    return "hip_singlethread";
  case SyncScope::HIPWavefront:
    return "hip_wavefront";
  case SyncScope::HIPWorkgroup:
    return "hip_workgroup";
  case SyncScope::HIPAgent:
    return "hip_agent";
  case SyncScope::HIPSystem:
    return "hip_system";
  case SyncScope::OpenCLWorkGroup:
    return "opencl_workgroup";
  case SyncScope::OpenCLDevice:
    return "opencl_device";
  case SyncScope::OpenCLAllSVMDevices:
    return "opencl_allsvmdevices";
  case SyncScope::OpenCLSubGroup:
    return "opencl_subgroup";
  }
  llvm_unreachable("Invalid synch scope");
}

/// Defines the kind of atomic scope models.
enum class AtomicScopeModelKind { None, OpenCL, HIP };

/// Defines the interface for synch scope model.
class AtomicScopeModel {
public:
  virtual ~AtomicScopeModel() {}
  /// Maps language specific synch scope values to internal
  /// SyncScope enum.
  virtual SyncScope map(unsigned S) const = 0;

  /// Check if the compile-time constant synch scope value
  /// is valid.
  virtual bool isValid(unsigned S) const = 0;

  /// Get all possible synch scope values that might be
  /// encountered at runtime for the current language.
  virtual ArrayRef<unsigned> getRuntimeValues() const = 0;

  /// If atomic builtin function is called with invalid
  /// synch scope value at runtime, it will fall back to a valid
  /// synch scope value returned by this function.
  virtual unsigned getFallBackValue() const = 0;

  /// Create an atomic scope model by AtomicScopeModelKind.
  /// \return an empty std::unique_ptr for AtomicScopeModelKind::None.
  static std::unique_ptr<AtomicScopeModel> create(AtomicScopeModelKind K);
};

/// Defines the synch scope model for OpenCL.
class AtomicScopeOpenCLModel : public AtomicScopeModel {
public:
  /// The enum values match the pre-defined macros
  /// __OPENCL_MEMORY_SCOPE_*, which are used to define memory_scope_*
  /// enums in opencl-c.h.
  enum ID {
    WorkGroup = 1,
    Device = 2,
    AllSVMDevices = 3,
    SubGroup = 4,
    Last = SubGroup
  };

  AtomicScopeOpenCLModel() {}

  SyncScope map(unsigned S) const override {
    switch (static_cast<ID>(S)) {
    case WorkGroup:
      return SyncScope::OpenCLWorkGroup;
    case Device:
      return SyncScope::OpenCLDevice;
    case AllSVMDevices:
      return SyncScope::OpenCLAllSVMDevices;
    case SubGroup:
      return SyncScope::OpenCLSubGroup;
    }
    llvm_unreachable("Invalid language synch scope value");
  }

  bool isValid(unsigned S) const override {
    return S >= static_cast<unsigned>(WorkGroup) &&
           S <= static_cast<unsigned>(Last);
  }

  ArrayRef<unsigned> getRuntimeValues() const override {
    static_assert(Last == SubGroup, "Does not include all synch scopes");
    static const unsigned Scopes[] = {
        static_cast<unsigned>(WorkGroup), static_cast<unsigned>(Device),
        static_cast<unsigned>(AllSVMDevices), static_cast<unsigned>(SubGroup)};
    return llvm::makeArrayRef(Scopes);
  }

  unsigned getFallBackValue() const override {
    return static_cast<unsigned>(AllSVMDevices);
  }
};

/// Defines the synch scope model for HIP.
class AtomicScopeHIPModel : public AtomicScopeModel {
public:
  /// The enum values match the pre-defined macros
  /// __HIP_MEMORY_SCOPE_*, which are used to define memory_scope_*
  /// enums in hip-c.h.
  enum ID {
    SingleThread = 1,
    Wavefront = 2,
    Workgroup = 3,
    Agent = 4,
    System = 5,
    Last = System
  };

  AtomicScopeHIPModel() {}

  SyncScope map(unsigned S) const override {
    switch (static_cast<ID>(S)) {
    case SingleThread:
      return SyncScope::HIPSingleThread;
    case Wavefront:
      return SyncScope::HIPWavefront;
    case Workgroup:
      return SyncScope::HIPWorkgroup;
    case Agent:
      return SyncScope::HIPAgent;
    case System:
      return SyncScope::HIPSystem;
    }
    llvm_unreachable("Invalid language synch scope value");
  }

  bool isValid(unsigned S) const override {
    return S >= static_cast<unsigned>(SingleThread) &&
           S <= static_cast<unsigned>(Last);
  }

  ArrayRef<unsigned> getRuntimeValues() const override {
    static_assert(Last == System, "Does not include all synch scopes");
    static const unsigned Scopes[] = {
        static_cast<unsigned>(SingleThread), static_cast<unsigned>(Wavefront),
        static_cast<unsigned>(Workgroup), static_cast<unsigned>(Agent),
        static_cast<unsigned>(System)};
    return llvm::makeArrayRef(Scopes);
  }

  unsigned getFallBackValue() const override {
    return static_cast<unsigned>(System);
  }
};

inline std::unique_ptr<AtomicScopeModel>
AtomicScopeModel::create(AtomicScopeModelKind K) {
  switch (K) {
  case AtomicScopeModelKind::None:
    return std::unique_ptr<AtomicScopeModel>{};
  case AtomicScopeModelKind::OpenCL:
    return std::make_unique<AtomicScopeOpenCLModel>();
  case AtomicScopeModelKind::HIP:
    return std::make_unique<AtomicScopeHIPModel>();
  }
  llvm_unreachable("Invalid atomic scope model kind");
}
} // namespace clang

#endif
