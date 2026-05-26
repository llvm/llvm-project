<!--===- docs/CUDA.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# CUDA Fortran

```{contents}
---
local:
---
```

Implementation notes for Flang's CUDA Fortran support.

## Lowering decisions

List of CUDA Fortran lowering decisions in Flang for cases where CUDA
Fortran interoperability requires behavior that is not specified by the
[CUDA Fortran Programming
Guide](https://docs.nvidia.com/hpc-sdk/compilers/cuda-fortran-prog-guide/index.html)
or by standard `BIND(C)` lowering alone.

### `BIND(C) ATTRIBUTES(GLOBAL)` assumed-shape and assumed-rank dummies

For a `BIND(C)` procedure with `ATTRIBUTES(GLOBAL)` or
`ATTRIBUTES(GRID_GLOBAL)`, an assumed-shape (`dimension(:)`) or assumed-rank
(`dimension(..)`) dummy is passed by base address (`!fir.ref<T>`) instead of by
`CFI_cdesc_t *` (`!fir.box<T>`). `ALLOCATABLE` and `POINTER` dummies take an
earlier descriptor-of-mutable path and are unaffected. A kernel declared
without `BIND(C)` (i.e., a plain `ATTRIBUTES(GLOBAL)` or
`ATTRIBUTES(GRID_GLOBAL)` Fortran kernel) uses the standard
descriptor-passing lowering instead, which is the form to use when the
kernel needs to access a CFI descriptor.

```fortran
interface
  attributes(global) subroutine f(d, n) bind(c, name='f')
    use iso_c_binding
    real(c_float), dimension(:), device :: d
    integer, value :: n
  end subroutine
end interface
! interoperates with: extern "C" __global__ void f(float *d, int n);
```

Reason: Fortran 2018 §18.3.7 prescribes a CFI descriptor for these shape
attributes under `BIND(C)`, but the
[`cudaLaunchKernel`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html)
ABI requires `args[i]` to point to a value of the type the C kernel declares
for parameter `i`. A descriptor pointer in `args[0]` would be dereferenced as
`T *` on the device: the kernel reads descriptor metadata bytes (`base_addr`,
`elem_len`, dim info, ...) as element data, producing wrong results, and when
the descriptor resides in host memory the device load additionally faults with
an illegal-access error.
