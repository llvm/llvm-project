//===- AddressSpaces.h - Language-specific address spaces -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Provides definitions for the various language-specific address
/// spaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ADDRESSSPACES_H
#define LLVM_CLANG_BASIC_ADDRESSSPACES_H

#include <array>
#include <cassert>
#include <initializer_list>
#include <utility>

namespace clang {

/// Defines the address space values used by the address space qualifier
/// of QualType.
///
enum class LangAS : unsigned {
  // The default value 0 is the value used in QualType for the situation
  // where there is no address space qualifier.
  Default = 0,

  // OpenCL specific address spaces.
  // In OpenCL each l-value must have certain non-default address space, each
  // r-value must have no address space (i.e. the default address space). The
  // pointee of a pointer must have non-default address space.
  opencl_global,
  opencl_local,
  opencl_constant,
  opencl_private,
  opencl_generic,
  opencl_global_device,
  opencl_global_host,

  // CUDA specific address spaces.
  cuda_device,
  cuda_constant,
  cuda_shared,

  // SYCL specific address spaces.
  sycl_global,
  sycl_global_device,
  sycl_global_host,
  sycl_local,
  sycl_private,

  // Pointer size and extension address spaces.
  ptr32_sptr,
  ptr32_uptr,
  ptr64,

  // HLSL specific address spaces.
  hlsl_groupshared,
  hlsl_constant,
  hlsl_private,
  hlsl_device,
  hlsl_input,
  hlsl_output,
  hlsl_push_constant,

  // Wasm specific address spaces.
  wasm_funcref,

  // AMDGPU address spaces
  amdgpu_barrier,

  // This denotes the count of language-specific address spaces and also
  // the offset added to the target-specific address spaces, which are usually
  // specified by address space attributes __attribute__(address_space(n))).
  FirstTargetAddressSpace
};

/// The type of a lookup table which maps from language-specific address spaces
/// to target-specific ones.
class LangASMap {
  std::array<unsigned, (unsigned)LangAS::FirstTargetAddressSpace> Map{};

public:
  constexpr LangASMap() = default;

  constexpr LangASMap(
      std::initializer_list<std::pair<LangAS, unsigned>> Mappings) {
    for (auto [LanguageAS, TargetAS] : Mappings)
      Map[(unsigned)LanguageAS] = TargetAS;
  }

  constexpr unsigned operator[](LangAS AS) const { return Map[(unsigned)AS]; }
};

/// \return whether \p AS is a target-specific address space rather than a
/// clang AST address space
inline bool isTargetAddressSpace(LangAS AS) {
  return (unsigned)AS >= (unsigned)LangAS::FirstTargetAddressSpace;
}

inline unsigned toTargetAddressSpace(LangAS AS) {
  assert(isTargetAddressSpace(AS));
  return (unsigned)AS - (unsigned)LangAS::FirstTargetAddressSpace;
}

inline LangAS getLangASFromTargetAS(unsigned TargetAS) {
  return static_cast<LangAS>((TargetAS) +
                             (unsigned)LangAS::FirstTargetAddressSpace);
}

inline bool isPtrSizeAddressSpace(LangAS AS) {
  return (AS == LangAS::ptr32_sptr || AS == LangAS::ptr32_uptr ||
          AS == LangAS::ptr64);
}

namespace PointeeAddressSpace {

enum ID : unsigned {
  Default = 0,
  OpenCLGlobal = 1,
  OpenCLLocal = 2,
  OpenCLConstant = 3,
  OpenCLPrivate = 4,
  OpenCLGeneric = 5,
  OpenCLGlobalDevice = 6,
  OpenCLGlobalHost = 7,
  CUDADevice = 8,
  CUDAConstant = 9,
  CUDAShared = 10,
  SYCLGlobal = 11,
  SYCLGlobalDevice = 12,
  SYCLGlobalHost = 13,
  SYCLLocal = 14,
  SYCLPrivate = 15,
  Ptr32Sptr = 16,
  Ptr32Uptr = 17,
  Ptr64 = 18,
  HLSLGroupShared = 19,
  HLSLConstant = 20,
  HLSLPrivate = 21,
  HLSLDevice = 22,
  HLSLInput = 23,
  HLSLOutput = 24,
  HLSLPushConstant = 25,
  WasmFuncRef = 26,
  HIPDevice = 27,
  HIPConstant = 28,
  HIPShared = 29,

  TargetOffset = 0x1000000
};

inline unsigned encode(LangAS AS, bool IsHIP = false) {
  if (isTargetAddressSpace(AS))
    return TargetOffset + toTargetAddressSpace(AS);

  switch (AS) {
  case LangAS::Default:
    return Default;
  case LangAS::opencl_global:
    return OpenCLGlobal;
  case LangAS::opencl_local:
    return OpenCLLocal;
  case LangAS::opencl_constant:
    return OpenCLConstant;
  case LangAS::opencl_private:
    return OpenCLPrivate;
  case LangAS::opencl_generic:
    return OpenCLGeneric;
  case LangAS::opencl_global_device:
    return OpenCLGlobalDevice;
  case LangAS::opencl_global_host:
    return OpenCLGlobalHost;
  case LangAS::cuda_device:
    return IsHIP ? HIPDevice : CUDADevice;
  case LangAS::cuda_constant:
    return IsHIP ? HIPConstant : CUDAConstant;
  case LangAS::cuda_shared:
    return IsHIP ? HIPShared : CUDAShared;
  case LangAS::sycl_global:
    return SYCLGlobal;
  case LangAS::sycl_global_device:
    return SYCLGlobalDevice;
  case LangAS::sycl_global_host:
    return SYCLGlobalHost;
  case LangAS::sycl_local:
    return SYCLLocal;
  case LangAS::sycl_private:
    return SYCLPrivate;
  case LangAS::ptr32_sptr:
    return Ptr32Sptr;
  case LangAS::ptr32_uptr:
    return Ptr32Uptr;
  case LangAS::ptr64:
    return Ptr64;
  case LangAS::hlsl_groupshared:
    return HLSLGroupShared;
  case LangAS::hlsl_constant:
    return HLSLConstant;
  case LangAS::hlsl_private:
    return HLSLPrivate;
  case LangAS::hlsl_device:
    return HLSLDevice;
  case LangAS::hlsl_input:
    return HLSLInput;
  case LangAS::hlsl_output:
    return HLSLOutput;
  case LangAS::hlsl_push_constant:
    return HLSLPushConstant;
  case LangAS::wasm_funcref:
    return WasmFuncRef;
  case LangAS::FirstTargetAddressSpace:
    break;
  }

  assert(false && "unknown language address space");
  return Default;
}

} // namespace PointeeAddressSpace

} // namespace clang

#endif // LLVM_CLANG_BASIC_ADDRESSSPACES_H
