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

The global variables `ResourceDescriptorHeap` and `SamplerDescriptorHeap` are
declared in the built-in header `hlsl/hlsl_resources.h`, which is included by
the default header `hlsl.h`. Indexing either global variable returns a small
internal struct that carries the heap index.

Indexing `ResourceDescriptorHeap` returns `__heap_resource_info`, and indexing
`SamplerDescriptorHeap` returns `__heap_sampler_info`. These structs are
defined by `HLSLExternalSemaSource` in the `hlsl::__detail` namespace. Using
distinct struct types allows the compiler to diagnose heap mismatches during
overload resolution: a sampler type can only be constructed from
`__heap_sampler_info`, while a CBV/SRV/UAV resource type can only be constructed
from `__heap_resource_info`.

Every resource class has an implicit constructor that accepts the
corresponding heap info struct. The constructor passes this information to the
Clang built-in function `__builtin_hlsl_resource_handlefromheap`, which creates
a concrete handle from the heap. Standard C++ implicit conversion rules allow
assignment, initialization, and function arguments to work without special
handling in `Sema`. For resources with counters, the constructor also uses
`__builtin_hlsl_resource_handlefromheap` to create the counter handle.

During code generation, the built-in function is lowered to target-specific
intrinsics for heap resources.

[dynamicresources]: https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html