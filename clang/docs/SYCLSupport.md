# SYCL Compiler and Runtime architecture design


## Introduction

This document describes the architecture of the SYCL compiler and runtime
library. More details are provided in
[external document](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/CompilerAndRuntimeDesign.md),
which are going to be added to clang documentation in the future.

## Predefined Macros

| Macro | Description |
| --- | --- |
| `__CLANG_RDC__` | Defined when Clang is compiling code in Relocatable Device Code (RDC) mode. RDC is necessary for linking device code across translation units. It is enabled by default for SYCL and can be disabled with the `-fno-gpu-rdc` compiler option. |

## Address space handling

The SYCL specification represents pointers to disjoint memory regions using C++
wrapper classes on an accelerator to enable compilation with a standard C++
toolchain and a SYCL compiler toolchain. Section 3.8.2 of SYCL 2020
specification defines
[memory model](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_device_memory_model),
section 4.7.7 - [address space classes](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_address_space_classes)
and section 5.9 covers [address space deduction](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_address_space_deduction).
The SYCL specification allows two modes of address space deduction: "generic as
default address space" (see section 5.9.3) and "inferred address space" (see
section 5.9.4). Current implementation supports only "generic as default address
space" mode.

SYCL borrows its memory model from OpenCL however SYCL doesn't perform
the address space qualifier inference as detailed in
[OpenCL C v3.0 6.7.8](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_C.html#addr-spaces-inference).

The default address space is "generic-memory", which is a virtual address space
that overlaps the global, local, and private address spaces. SYCL mode enables
following conversions:

- explicit conversions to/from the default address space from/to the address
  space-attributed type
- implicit conversions from the address space-attributed type to the default
  address space
- explicit conversions to/from the global address space from/to the
  `__attribute__((opencl_global_device))` or
  `__attribute__((opencl_global_host))` address space-attributed type
- implicit conversions from the `__attribute__((opencl_global_device))` or
  `__attribute__((opencl_global_host))` address space-attributed type to the
  global address space

All named address spaces are disjoint and sub-sets of default address space.

The SPIR target allocates SYCL namespace scope variables in the global address
space.

Pointers to default address space should get lowered into a pointer to a generic
address space (or flat to reuse more general terminology). But depending on the
allocation context, the default address space of a non-pointer type is assigned
to a specific address space. This is described in
[common address space deduction rules](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:commonAddressSpace)
section.

This is also in line with the behaviour of CUDA ([small example](https://godbolt.org/z/veqTfo9PK)).

`multi_ptr` class implementation example:

```C++
// check that SYCL mode is ON and we can use non-standard decorations
#if defined(__SYCL_DEVICE_ONLY__)
// GPU/accelerator implementation
template <typename T, address_space AS> class multi_ptr {
  // DecoratedType applies corresponding address space attribute to the type T
  // DecoratedType<T, global_space>::type == "__attribute__((opencl_global)) T"
  // See sycl/include/CL/sycl/access/access.hpp for more details
  using pointer_t = typename DecoratedType<T, AS>::type *;

  pointer_t m_Pointer;
  public:
  pointer_t get() { return m_Pointer; }
  T& operator* () { return *reinterpret_cast<T*>(m_Pointer); }
}
#else
// CPU/host implementation
template <typename T, address_space AS> class multi_ptr {
  T *m_Pointer; // regular undecorated pointer
  public:
  T *get() { return m_Pointer; }
  T& operator* () { return *m_Pointer; }
}
#endif
```

Depending on the compiler mode, `multi_ptr` will either decorate its internal
data with the address space attribute or not.

To utilize clang's existing functionality, we reuse the following OpenCL address
space attributes for pointers:

| Address space attribute                         | SYCL address_space enumeration |
| ----------------------------------------------- | ------------------------------ |
| `__attribute__((opencl_global))`                | global_space, constant_space   |
| `__attribute__((opencl_global_device))`         | global_space                   |
| `__attribute__((opencl_global_host))`           | global_space                   |
| `__attribute__((opencl_local))`                 | local_space                    |
| `__attribute__((opencl_private))`               | private_space                  |

```C++
//TODO: add support for __attribute__((opencl_global_host)) and __attribute__((opencl_global_device)).
```

## Separate device link

`-fsycl-link` links the SYCL device code of its inputs and wraps the resulting
device image into a host object. The host linker is not run. The object is
then used in a later host link, which can be done by any host linker.

```console
clang++ -fsycl -c a.cpp b.cpp
clang++ -fsycl -fsycl-link a.o b.o -o dev.o
clang++ -c main.cpp
clang++ main.o a.o b.o dev.o -L<libsycl dir> -lLLVMSYCL -o app
```

Here `main.cpp` has no SYCL device code. Device code that is not in the
inputs of `-fsycl-link` is not in `dev.o`.

On Windows, the host link uses `LLVMSYCL.lib` instead of `-lLLVMSYCL`. Without
`-o`, the output is named as for a host link, e.g. `a.out` or `a.exe`.

The host link does not use `-fsycl`. With `-fsycl`, the device code in `a.o`
and `b.o` would be linked and registered a second time.

If the inputs contain no device code, the output is an empty object without
SYCL runtime registration.

Device code in a static library is only extracted for device targets that
the other inputs already have. If only static libraries provide device code,
pass them with `-Wl,--whole-archive libfoo.a -Wl,--no-whole-archive`, or with
`-Wl,/wholearchive:foo.lib` on Windows.

The output of `-fsycl-link` is not yet supported in a static library: nothing
in it is referenced by the host code, so the host linker does not pull it out
of the archive and the device images are never registered.
