//===----- hlsl_resources.h - HLSL definitions for resources ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_RESOURCES_H_
#define _HLSL_HLSL_RESOURCES_H_

#include "hlsl_detail.h"

namespace hlsl {

#define _HLSL_AVAILABILITY(platform, version)                                  \
  __attribute__((availability(platform, introduced = version)))

_HLSL_AVAILABILITY(shadermodel, 6.6)
static __detail::__resource_descriptor_heap_struct ResourceDescriptorHeap;

_HLSL_AVAILABILITY(shadermodel, 6.6)
static __detail::__sampler_descriptor_heap_struct SamplerDescriptorHeap;

} // namespace hlsl
#endif //_HLSL_HLSL_RESOURCES_H_
