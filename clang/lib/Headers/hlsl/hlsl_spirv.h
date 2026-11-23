//===----- hlsl_spirv.h - HLSL definitions for SPIR-V target --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_SPIRV_H_
#define _HLSL_HLSL_SPIRV_H_

namespace hlsl {
namespace vk {
template <typename T, T v> struct integral_constant {
  static constexpr T value = v;
};

template <typename T> struct Literal {};

template <uint Opcode, uint Size, uint Alignment, typename... Operands>
using SpirvType = __hlsl_spirv_type<Opcode, Size, Alignment, Operands...>;

template <uint Opcode, typename... Operands>
using SpirvOpaqueType = __hlsl_spirv_type<Opcode, 0, 0, Operands...>;
} // namespace vk
} // namespace hlsl

#endif // _HLSL_HLSL_SPIRV_H_
