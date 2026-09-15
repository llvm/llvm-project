# HLSL Dynamic Resources

```{contents}
:local: true
```

## Introduction

[HLSL Dynamic Resources][dynamicresources] is a feature introduced in Shader
Model 6.6 that allows shaders to create resources from descriptors by directly
indexing the CBV/SRV/UAV heap or the sampler heap. Dynamic resources are also
known as _bindless resources_ or _directly indexed resources_.

HLSL exposes these heaps through two built-in global objects:
`ResourceDescriptorHeap` and `SamplerDescriptorHeap`. Indexing either object
yields an intermediate value that can be converted to a resource or sampler
object. Resources created this way do not require binding locations or root
signature descriptor-table mappings.

## Implementation Details

The built-in header `hlsl/hlsl_resources.h` declares the global variables
`ResourceDescriptorHeap` and `SamplerDescriptorHeap` and is included by the
default header `hlsl.h`. Their types, `resource_descriptor_heap_struct` and
`sampler_descriptor_heap_struct`, are defined in the `hlsl::__detail` namespace
in `hlsl/hlsl_detail.h`.

Each global's indexing operator returns a small internal struct containing the
heap index: `ResourceDescriptorHeap` returns `heap_resource_info`, while
`SamplerDescriptorHeap` returns `heap_sampler_info`. `HLSLExternalSemaSource`
defines both structs in the `hlsl::__detail` namespace. Their distinct types
allow overload resolution to reject attempts to construct a resource from the
wrong heap: sampler types accept only `heap_sampler_info`, while CBV/SRV/UAV
resource types accept only `heap_resource_info`.

Each resource class has an implicit constructor that accepts the corresponding
heap info struct and passes it to `__builtin_hlsl_resource_handlefromheap` to
create a concrete resource handle. This allows assignments, initializations,
and function arguments to use standard C++ implicit conversions without special
handling in `Sema`. For resources with counters, the constructor creates the
counter handle with the separate
`__builtin_hlsl_resource_counterhandlefromheap` built-in.

During code generation, each call to the built-in is lowered to target-specific
intrinsics for heap resources.

[dynamicresources]: https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html