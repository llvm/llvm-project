```{contents}
:local: true
```

# AMDGPU Support

Clang supports OpenCL, HIP and OpenMP on AMD GPU targets.

## Predefined Macros

| Macro | Description |
| --- | --- |
| `__AMDGPU__` | Indicates that the code is being compiled for an AMD GPU. |
| `__AMDGCN__` | Defined if the GPU target is AMDGCN. |
| `__R600__` | Defined if the GPU target is R600. |
| `__<ArchName>__` | Defined with the name of the architecture (e.g., `__gfx906__` for the gfx906 architecture). |
| `__<GFXN>__` | Defines the GFX family (e.g., for gfx906, this macro would be `__GFX9__`). |
| `__amdgcn_processor__` | Defined with the processor name as a string (e.g., `"gfx906"`). |
| `__amdgcn_target_id__` | Defined with the target ID as a string. |
| `__amdgcn_feature_<feature-name>__` | Defined for each supported target feature. The value is 1 if the feature is enabled and 0 if it is disabled. Allowed feature names are sramecc and xnack. |
| `__AMDGCN_CUMODE__` | Defined as 1 if the CU mode is enabled and 0 if the WGP mode is enabled. |
| `__AMDGCN_UNSAFE_FP_ATOMICS__` | Defined if unsafe floating-point atomics are allowed. |
| `__HAS_FMAF__` | Defined if FMAF instruction is available (deprecated). |
| `__HAS_LDEXPF__` | Defined if LDEXPF instruction is available (deprecated). |
| `__HAS_FP64__` | Defined if FP64 instruction is available (deprecated). |

Please note that the specific architecture and feature names will vary depending on the GPU. Also, some macros are deprecated and may be removed in future releases.

## Target-Specific Builtins

Clang exposes AMDGPU hardware intrinsics as target-specific builtins with the
`__builtin_amdgcn_` prefix. These are documented in {doc}`AMDGPUBuiltinReference`.

## Target-Specific Types

### Named Workgroup Barrier Type

The `__amdgpu_named_workgroup_barrier_t` type is used to represent the GFX12.5 named barriers.
Example usage:

```c
__amdgpu_named_workgroup_barrier_t x;
__amdgpu_named_workgroup_barrier_t arr[2]; // Arrays are also fine

void foo(int a)
{
  __builtin_amdgcn_s_barrier_init(&x, a);
}
```

A "named barrier wrapper" is a class that contains exactly one field, which is either
a single value or an array of values of one of the following types:

* `__amdgpu_named_workgroup_barrier_t`.
* Another "named barrier wrapper".

Named barrier wrappers let users add helper methods around named barrier objects.

In C++, a class that inherits from a named barrier wrapper is also considered a
named barrier wrapper. Named barrier wrappers must be standard-layout types
(see `std::is_standard_layout`).
This means that named barrier wrappers:

* May not have a virtual table: they cannot declare or inherit any virtual
  functions, or inherit from a virtual base class.
* May not have any extra fields, either declared by the class or inherited
  from a base class.
