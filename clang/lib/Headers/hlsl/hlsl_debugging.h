//===----- hlsl_debugging.h - HLSL debugging functions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_DEBUGGING_H_
#define _HLSL_HLSL_DEBUGGING_H_

namespace hlsl {

_HLSL_AVAILABILITY(shadermodel, 6.10)
void inline DebugBreak() { __builtin_debugtrap(); }

} // namespace hlsl

#endif // _HLSL_HLSL_DEBUGGING_H_
