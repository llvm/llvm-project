## OpenCL Address Spaces

The address space qualifier may be used to specify the region of memory that is
used to allocate the object. OpenCL supports the following address spaces:
`__generic` (or `generic`), `__global` (or `global`), `__local` (or `local`),
`__private` (or `private`), and `__constant` (or `constant`).

```c
__constant int c = ...;

__generic int* foo(global int* g) {
  __local int* l;
  private int p;
  ...
  return l;
}
```

More details can be found in the OpenCL C language Specification version 3,
Section 6.7, "Address Space Qualifiers".

### [[clang::opencl_global_device]], [[clang::opencl_global_host]]

{clang-attr-syntaxes}`OpenCLAddressSpaceGlobalExtDocs`

The `global_device` and `global_host` address space attributes specify that
an object is allocated in global memory on the device/host. It helps to
distinguish USM (Unified Shared Memory) pointers that access global device
memory from those that access global host memory. These new address spaces are
a subset of the `__global/opencl_global` address space, the full address space
set model for OpenCL 2.0 with the extension looks as follows:

```text
generic->global->host
               ->device
       ->private
       ->local
constant
```

As `global_device` and `global_host` are a subset of
`__global/opencl_global` address spaces it is allowed to convert
`global_device` and `global_host` address spaces to
`__global/opencl_global` address spaces (following ISO/IEC TR 18037 5.1.3
"Address space nesting and rules for pointers").

These attributes are deprecated and may be removed in a future version of Clang.


### __constant, constant, [[clang::opencl_constant]]

{clang-attr-syntaxes}`OpenCLAddressSpaceConstantDocs`

The constant address space attribute signals that an object is located in
a constant (non-modifiable) memory region. It is available to all work items.
Any type can be annotated with the constant address space attribute. Objects
with the constant address space qualifier can be declared in any scope and must
have an initializer.


### __generic, generic, [[clang::opencl_generic]]

{clang-attr-syntaxes}`OpenCLAddressSpaceGenericDocs`

The generic address space attribute is only available with OpenCL v2.0 and later.
It can be used with pointer types. Variables in global and local scope and
function parameters in non-kernel functions can have the generic address space
type attribute. It is intended to be a placeholder for any other address space
except for `__constant` in OpenCL code which can be used with multiple address
spaces.


### __global, global, [[clang::opencl_global]]

{clang-attr-syntaxes}`OpenCLAddressSpaceGlobalDocs`

The global address space attribute specifies that an object is allocated in
global memory, which is accessible by all work items. The content stored in this
memory area persists between kernel executions. Pointer types to the global
address space are allowed as function parameters or local variables. Starting
with OpenCL v2.0, the global address space can be used with global (program
scope) variables and static local variable as well.


### __local, local, [[clang::opencl_local]]

{clang-attr-syntaxes}`OpenCLAddressSpaceLocalDocs`

The local address space specifies that an object is allocated in the local (work
group) memory area, which is accessible to all work items in the same work
group. The content stored in this memory region is not accessible after
the kernel execution ends. In a kernel function scope, any variable can be in
the local address space. In other scopes, only pointer types to the local address
space are allowed. Local address space variables cannot have an initializer.


### __private, private, [[clang::opencl_private]]

{clang-attr-syntaxes}`OpenCLAddressSpacePrivateDocs`

The private address space specifies that an object is allocated in the private
(work item) memory. Other work items cannot access the same memory area and its
content is destroyed after work item execution ends. Local variables can be
declared in the private address space. Function arguments are always in the
private address space. Kernel function arguments of a pointer or an array type
cannot point to the private address space.


