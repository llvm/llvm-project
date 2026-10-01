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
#define _HLSL_BUILTIN_ALIAS(builtin)                                           \
  __attribute__((clang_builtin_alias(builtin)))

static const uint UAV_MEMORY = 0x1;
static const uint GROUP_SHARED_MEMORY = 0x2;
static const uint NODE_INPUT_MEMORY = 0x4;
static const uint NODE_OUTPUT_MEMORY = 0x8;
static const uint ALL_MEMORY = 0xf;

static const uint GROUP_SYNC = 0x1;
static const uint GROUP_SCOPE = 0x2;
static const uint DEVICE_SCOPE = 0x4;

#define _HLSL_RESOURCE_BARRIER(resource)                                       \
  _HLSL_AVAILABILITY(shadermodel, 6.8)                                         \
  _HLSL_BUILTIN_ALIAS(__builtin_hlsl_barrier)                                  \
  __attribute__((convergent)) void Barrier(resource Object, uint SemanticFlags)

#define _HLSL_TEMPLATE_RESOURCE_BARRIER(resource)                              \
  template <typename T> _HLSL_RESOURCE_BARRIER(resource<T>)

_HLSL_TEMPLATE_RESOURCE_BARRIER(RWTexture1D);
_HLSL_TEMPLATE_RESOURCE_BARRIER(RWTexture1DArray);
_HLSL_TEMPLATE_RESOURCE_BARRIER(RWTexture2D);
_HLSL_TEMPLATE_RESOURCE_BARRIER(RWTexture2DArray);
_HLSL_TEMPLATE_RESOURCE_BARRIER(RWTexture3D);
_HLSL_TEMPLATE_RESOURCE_BARRIER(RWBuffer);
_HLSL_TEMPLATE_RESOURCE_BARRIER(RasterizerOrderedBuffer);
_HLSL_TEMPLATE_RESOURCE_BARRIER(RWStructuredBuffer);
_HLSL_TEMPLATE_RESOURCE_BARRIER(AppendStructuredBuffer);
_HLSL_TEMPLATE_RESOURCE_BARRIER(ConsumeStructuredBuffer);
_HLSL_TEMPLATE_RESOURCE_BARRIER(RasterizerOrderedStructuredBuffer);
_HLSL_RESOURCE_BARRIER(RWByteAddressBuffer);
_HLSL_RESOURCE_BARRIER(RasterizerOrderedByteAddressBuffer);

#undef _HLSL_TEMPLATE_RESOURCE_BARRIER
#undef _HLSL_RESOURCE_BARRIER

_HLSL_AVAILABILITY(shadermodel, 6.6)
static __detail::resource_descriptor_heap_struct ResourceDescriptorHeap;

_HLSL_AVAILABILITY(shadermodel, 6.6)
static __detail::sampler_descriptor_heap_struct SamplerDescriptorHeap;

} // namespace hlsl
#endif //_HLSL_HLSL_RESOURCES_H_
