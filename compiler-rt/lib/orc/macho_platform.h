//===- macho_platform.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ORC Runtime support for Darwin dynamic loading features.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_MACHO_PLATFORM_H
#define ORC_RT_MACHO_PLATFORM_H

#include "common.h"
#include "executor_address.h"

// Atexit functions.
ORC_RT_INTERFACE int __orc_rt_macho_cxa_atexit(void (*func)(void *), void *arg,
                                               void *dso_handle);
ORC_RT_INTERFACE void __orc_rt_macho_cxa_finalize(void *dso_handle);

// dlfcn functions.
ORC_RT_INTERFACE const char *__orc_rt_macho_jit_dlerror();
ORC_RT_INTERFACE void *__orc_rt_macho_jit_dlopen(const char *path, int mode);
ORC_RT_INTERFACE int __orc_rt_macho_jit_dlclose(void *dso_handle);
ORC_RT_INTERFACE void *__orc_rt_macho_jit_dlsym(void *dso_handle,
                                                const char *symbol);

namespace __orc_rt {
namespace macho {

struct MachOPerObjectSectionsToRegister {
  ExecutorAddressRange EHFrameSection;
  ExecutorAddressRange ThreadDataSection;
};

struct MachOJITDylibInitializers {
  using SectionList = std::vector<ExecutorAddressRange>;

  MachOJITDylibInitializers() = default;
  MachOJITDylibInitializers(std::string Name,
                            ExecutorAddress MachOHeaderAddress)
      : Name(std::move(Name)),
        MachOHeaderAddress(std::move(MachOHeaderAddress)) {}

  std::string Name;
  ExecutorAddress MachOHeaderAddress;
  ExecutorAddress ObjCImageInfoAddress;

  std::unordered_map<std::string, SectionList> InitSections;
};

class MachOJITDylibDeinitializers {};

using MachOJITDylibInitializerSequence = std::vector<MachOJITDylibInitializers>;

using MachOJITDylibDeinitializerSequence =
    std::vector<MachOJITDylibDeinitializers>;

enum dlopen_mode : int {
  ORC_RT_RTLD_LAZY = 0x1,
  ORC_RT_RTLD_NOW = 0x2,
  ORC_RT_RTLD_LOCAL = 0x4,
  ORC_RT_RTLD_GLOBAL = 0x8
};

} // end namespace macho

using SPSMachOPerObjectSectionsToRegister =
    SPSTuple<SPSExecutorAddressRange, SPSExecutorAddressRange>;

template <>
class SPSSerializationTraits<SPSMachOPerObjectSectionsToRegister,
                             macho::MachOPerObjectSectionsToRegister> {

public:
  static size_t size(const macho::MachOPerObjectSectionsToRegister &MOPOSR) {
    return SPSMachOPerObjectSectionsToRegister::AsArgList::size(
        MOPOSR.EHFrameSection, MOPOSR.ThreadDataSection);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const macho::MachOPerObjectSectionsToRegister &MOPOSR) {
    return SPSMachOPerObjectSectionsToRegister::AsArgList::serialize(
        OB, MOPOSR.EHFrameSection, MOPOSR.ThreadDataSection);
  }

  static bool deserialize(SPSInputBuffer &IB,
                          macho::MachOPerObjectSectionsToRegister &MOPOSR) {
    return SPSMachOPerObjectSectionsToRegister::AsArgList::deserialize(
        IB, MOPOSR.EHFrameSection, MOPOSR.ThreadDataSection);
  }
};

using SPSNamedExecutorAddressRangeSequenceMap =
    SPSSequence<SPSTuple<SPSString, SPSExecutorAddressRangeSequence>>;

using SPSMachOJITDylibInitializers =
    SPSTuple<SPSString, SPSExecutorAddress, SPSExecutorAddress,
             SPSNamedExecutorAddressRangeSequenceMap>;

using SPSMachOJITDylibInitializerSequence =
    SPSSequence<SPSMachOJITDylibInitializers>;

/// Serialization traits for MachOJITDylibInitializers.
template <>
class SPSSerializationTraits<SPSMachOJITDylibInitializers,
                             macho::MachOJITDylibInitializers> {
public:
  static size_t size(const macho::MachOJITDylibInitializers &MOJDIs) {
    return SPSMachOJITDylibInitializers::AsArgList::size(
        MOJDIs.Name, MOJDIs.MachOHeaderAddress, MOJDIs.ObjCImageInfoAddress,
        MOJDIs.InitSections);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const macho::MachOJITDylibInitializers &MOJDIs) {
    return SPSMachOJITDylibInitializers::AsArgList::serialize(
        OB, MOJDIs.Name, MOJDIs.MachOHeaderAddress, MOJDIs.ObjCImageInfoAddress,
        MOJDIs.InitSections);
  }

  static bool deserialize(SPSInputBuffer &IB,
                          macho::MachOJITDylibInitializers &MOJDIs) {
    return SPSMachOJITDylibInitializers::AsArgList::deserialize(
        IB, MOJDIs.Name, MOJDIs.MachOHeaderAddress, MOJDIs.ObjCImageInfoAddress,
        MOJDIs.InitSections);
  }
};

} // end namespace __orc_rt

#endif // ORC_RT_MACHO_PLATFORM_H
