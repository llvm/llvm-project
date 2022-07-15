//===- HLSLRuntime.h - HLSL Runtime -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the clang::IdentifierInfo, clang::IdentifierTable, and
/// clang::Selector interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_BASIC_HLSLRUNTIME_H
#define CLANG_BASIC_HLSLRUNTIME_H

#include <cstdint>

namespace clang {
namespace hlsl {

enum class ResourceClass : uint8_t {
  SRV = 0,
  UAV,
  CBuffer,
  Sampler,
  NumClasses
};

} // namespace hlsl
} // namespace clang

#endif // CLANG_BASIC_HLSLRUNTIME_H
