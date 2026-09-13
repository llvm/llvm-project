//===----- hlsl_resources.h - HLSL definitions for resources ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_RESOURCES_H_
#define _HLSL_HLSL_RESOURCES_H_

namespace hlsl {

#define _HLSL_AVAILABILITY(platform, version)                                  \
  __attribute__((availability(platform, introduced = version)))

struct __hlsl_resource_descriptor_heap_struct {
  __hlsl_heap_resource_info operator[](uint32_t Index) {
    return __hlsl_heap_resource_info{Index};
  }
};

struct __hlsl_sampler_descriptor_heap_struct {
  __hlsl_heap_sampler_info operator[](uint32_t Index) {
    return __hlsl_heap_sampler_info{Index};
  }
};

_HLSL_AVAILABILITY(shadermodel, 6.6)
static __hlsl_resource_descriptor_heap_struct ResourceDescriptorHeap;

_HLSL_AVAILABILITY(shadermodel, 6.6)
static __hlsl_sampler_descriptor_heap_struct SamplerDescriptorHeap;

} // namespace hlsl
#endif //_HLSL_HLSL_RESOURCES_H_
