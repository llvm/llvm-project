## Type Attributes



### SYCL Address Spaces

{clang-attr-syntaxes}`SYCLAddressSpaceDocs`

:::{Note}
These attributes are intended for use in the implementation of SYCL run-time
libraries and should not be used in any other context.
Programmers writing code intended to conform to the SYCL specification should
use the address space facilities specified in the following sections of the
SYCL 2020 specification.

* [4.7.2, "Buffers"][SYCL-2020-4.7.2]
* [4.7.6, "Accessors"][SYCL-2020-4.7.6]
* [4.7.7, "Address space classes"][SYCL-2020-4.7.7]
* [F.7, "sycl_khr_static_addrspace_cast"][SYCL-2020-F.7]
* [F.8, "sycl_khr_dynamic_addrspace_cast"][SYCL-2020-F.8]
:::

The SYCL address space attributes listed below correspond to the five address
spaces described by
[SYCL 2020 section 3.8.2, "SYCL device memory model"][SYCL-2020-3.8.2] and
[SYCL 2020 section 4.7.7, "Address space classes"][SYCL-2020-4.7.7].

::: {list-table} SYCL address space attributes
:header-rows: 1

* - Address space attribute
  - SYCL address space
  - Description
* - `[[clang::sycl_global]]`
  - global
  - A memory region accessible by all work-items executing on a device.
* - `[[clang::sycl_local]]`
  - local
  - A memory region accessible by all work-items of a single work-group.
* - `[[clang::sycl_private]]`
  - private
  - A memory region that is private to a single work-item.
* - `[[clang::sycl_generic]]`
  - generic
  - A virtual memory region from which the global, local, and private memory
    regions may all be accessed.
* - `[[clang::sycl_constant]]`
  - constant
  - (*deprecated*) A memory region that holds constant data for an executing
    kernel.
:::

The SYCL address space attributes are type attributes that may be applied to
non-function non-reference types to specify an address space qualified type.

A type with a SYCL address space qualifier is a distinct type from the
otherwise unattributed type. For example, `int *` and `int [[clang::sycl_global]]*`
designate distinct pointer types which participate in overload resolution and
template specialization.

The top-level type of a variable declaration cannot have a SYCL address space
qualifier. For example:

```c++
int [[clang::sycl_global]] gv;   // error: the top-level type has an address space qualifier.
int [[clang::sycl_global]] *pgi; // ok; the address space qualifier is on the pointee type.
```

Conversions between SYCL address space attributed types are permitted as
follows.

- Types attributed with the global, local, or private address space attributes
  are implicitly convertible to matching types with the generic address space
  attribute.

The mapping of SYCL address spaces to physical address spaces is target
dependent.

For OpenCL device targets, the SYCL address space attributes are aligned with
the [OpenCL address space attributes](#opencl-address-spaces) such that, e.g.,
`int [[clang::sycl_global]]*` and `int [[clang::opencl_global]]*` specify
distinct types both of which map to the same underlying address space.
Corresponding SYCL and OpenCL address space attributed types are implicitly
convertible; other conversions are permitted as described above; e.g.,
`int [[clang::sycl_global]]*` is implicitly convertible to
`int [[clang::opencl_generic]]*`.

[SYCL-2020-3.8.2]: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_device_memory_model
[SYCL-2020-4.7.2]: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:buffers
[SYCL-2020-4.7.6]: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:accessors
[SYCL-2020-4.7.7]: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_address_space_classes
[SYCL-2020-F.7]: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:khr-static-addrspace-cast
[SYCL-2020-F.8]: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:khr-dynamic-addrspace-cast


### __ptr32

{clang-attr-syntaxes}`Ptr32Docs`

The `__ptr32` qualifier represents a native pointer on a 32-bit system. On a
64-bit system, a pointer with `__ptr32` is extended to a 64-bit pointer. The
`__sptr` and `__uptr` qualifiers can be used to specify whether the pointer
is sign extended or zero extended. This qualifier is enabled under
`-fms-extensions`.


### __ptr64

{clang-attr-syntaxes}`Ptr64Docs`

The `__ptr64` qualifier represents a native pointer on a 64-bit system. On a
32-bit system, a `__ptr64` pointer is truncated to a 32-bit pointer. This
qualifier is enabled under `-fms-extensions`.


### __sptr

{clang-attr-syntaxes}`SPtrDocs`

The `__sptr` qualifier specifies that a 32-bit pointer should be sign
extended when converted to a 64-bit pointer.


### __uptr

{clang-attr-syntaxes}`UPtrDocs`

The `__uptr` qualifier specifies that a 32-bit pointer should be zero
extended when converted to a 64-bit pointer.


(langext-address_space_documentation)=

### address_space

{clang-attr-syntaxes}`AddressSpaceDocs`

:::{Note}
This attribute is mainly intended to be used by target headers
provided by the toolchain. End users should prefer the documented, named
address space annotations for their platform, such as the
[OpenCL address spaces](#opencl-address-spaces), `__global__`, `__local__`,
or something else.
:::

The `address_space` attribute functions as a type qualifier that allows the
programmer to specify the address space for a pointer or reference type.
Qualified pointer types are considered distinct types for the purposes of
overload resolution. The attribute takes a single, non-negative integer
constant expression identifying the address space. For example:

```c
int * __attribute__((address_space(1))) ptr;

void foo(__attribute__((address_space(2))) float *buf);
```

Only one address space qualifier may be applied to a given pointer or reference
type. Where address spaces are allowed (e.g., variables, parameters, return
types) and what values are valid depends on the target and language mode.

The meaning of each value is defined by the target; multiple address spaces are
used in environments such as OpenCL, CUDA, HIP, and other GPU programming
models to distinguish global, local, constant, and private memory. See for
example the address spaces defined in the [NVPTX User Guide][nvptx user guide] and the
[AMDGPU User Guide][amdgpu user guide].

Address spaces may partially overlap or be entirely distinct. The compiler may
reject attempts to convert between distinct, incompatible address spaces.
Pointer width may vary between different address spaces, so some explicit casts
may truncate.

For more information, refer to [ISO TR18037][iso tr18037], which covers embedded C language
extensions. Section 5 covers named address spaces.

[amdgpu user guide]: https://llvm.org/docs/AMDGPUUsage.html#address-spaces
[iso tr18037]: https://standards.iso.org/ittf/PubliclyAvailableStandards/c051126_ISO_IEC_TR_18037_2008.zip
[nvptx user guide]: https://llvm.org/docs/NVPTXUsage.html#address-spaces


### align_value

{clang-attr-syntaxes}`AlignValueDocs`

The align_value attribute can be added to the typedef of a pointer type or the
declaration of a variable of pointer or reference type. It specifies that the
pointer will point to, or the reference will bind to, only objects with at
least the provided alignment. This alignment value must be some positive power
of 2.

```c
typedef double * aligned_double_ptr __attribute__((align_value(64)));
void foo(double & x  __attribute__((align_value(128))),
         aligned_double_ptr y) { ... }
```

If the pointer value does not have the specified alignment at runtime, the
behavior of the program is undefined.


### annotate_type

{clang-attr-syntaxes}`AnnotateTypeDocs`

This attribute is used to add annotations to types, typically for use by static
analysis tools that are not integrated into the core Clang compiler (e.g.,
Clang-Tidy checks or out-of-tree Clang-based tools). It is a counterpart to the
`annotate` attribute, which serves the same purpose, but for declarations.

The attribute takes a mandatory string literal argument specifying the
annotation category and an arbitrary number of optional arguments that provide
additional information specific to the annotation category. The optional
arguments must be constant expressions of arbitrary type.

For example:

```c++
int* [[clang::annotate_type("category1", "foo", 1)]] f(int[[clang::annotate_type("category2")]] *);
```

The attribute does not have any effect on the semantics of the type system,
neither type checking rules, nor runtime semantics. In particular:

- `std::is_same<T, T [[clang::annotate_type("foo")]]>` is true for all types
  `T`.
- It is not permissible for overloaded functions or template specializations
  to differ merely by an `annotate_type` attribute.
- The presence of an `annotate_type` attribute will not affect name
  mangling.


### arm_sve_vector_bits

{clang-attr-syntaxes}`ArmSveVectorBitsDocs`

The `arm_sve_vector_bits(N)` attribute is defined by the Arm C Language
Extensions (ACLE) for SVE. It is used to define fixed-length (VLST) variants of
sizeless types (VLAT).

For example:

```c
#include <arm_sve.h>

#if __ARM_FEATURE_SVE_BITS==512
typedef svint32_t fixed_svint32_t __attribute__((arm_sve_vector_bits(512)));
#endif
```

Creates a type `fixed_svint32_t` that is a fixed-length variant of
`svint32_t` that contains exactly 512-bits. Unlike `svint32_t`, this type
can be used in globals, structs, unions, and arrays, all of which are
unsupported for sizeless types.

The attribute can be attached to a single SVE vector (such as `svint32_t`) or
to the SVE predicate type `svbool_t`, this excludes tuple types such as
`svint32x4_t`. The behavior of the attribute is undefined unless
`N==__ARM_FEATURE_SVE_BITS`, the implementation defined feature macro that is
enabled under the `-msve-vector-bits` flag.

For more information See [Arm C Language Extensions for SVE](https://developer.arm.com/documentation/100987/latest) for more information.


### bpf_fastcall

{clang-attr-syntaxes}`BPFFastCallDocs`

Functions annotated with this attribute are likely to be inlined by BPF JIT.
It is assumed that inlined implementation uses less caller saved registers,
than a regular function.
Specifically, the following registers are likely to be preserved:
- `R0` if function return value is `void`;
- `R2-R5` if function takes 1 argument;
- `R3-R5` if function takes 2 arguments;
- `R4-R5` if function takes 3 arguments;
- `R5` if function takes 4 arguments;

For such functions Clang generates code pattern that allows BPF JIT
to recognize and remove unnecessary spills and fills of the preserved
registers.


### btf_type_tag

{clang-attr-syntaxes}`BTFTypeTagDocs`

Clang supports the `__attribute__((btf_type_tag("ARGUMENT")))` attribute for
all targets. It only has effect when `-g` is specified on the command line.

The attribute can be applied to a pointer type, in which case the tag is
associated with the pointee type, e.g.:

```c
int __attribute__((btf_type_tag("tag"))) *p;
```

It can also be applied to the underlying type of a typedef, in which case the
tag follows the typedef down to its base type, e.g.:

```c
typedef struct foo __attribute__((btf_type_tag("tag"))) foo_t;
```

The following is the corresponding btf:

```
...
[2] TYPE_TAG 'tag' type_id=4
[3] TYPEDEF 'foo_t' type_id=2
[4] STRUCT 'foo' size=4 vlen=1
    'c' type_id=5 bits_offset=0
[5] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
...
```

The attribute is currently silently ignored in any other position (note: this
scenario may be diagnosed in the future).

The `ARGUMENT` string will be preserved in IR and emitted to DWARF for the
types used in variable declarations, function declarations, or typedef
declarations.

For BPF targets, the `ARGUMENT` string will also be emitted to .BTF ELF
section.


### cfi_unchecked_callee

{clang-attr-syntaxes}`CFIUncheckedCalleeDocs`

`cfi_unchecked_callee` is a function type attribute which prevents the
compiler from instrumenting
{doc}`Control Flow Integrity <ControlFlowIntegrity>` checks on indirect
function calls. This also includes control flow checks added by
`-fsanitize=function`; see {ref}`Available checks <ubsan-checks>`.
Specifically, the attribute has the following semantics:

1. Indirect calls to a function type with this attribute will not be instrumented with CFI. That is,
   the indirect call will not be checked. Note that this only changes the behavior for indirect calls
   on pointers to function types having this attribute. It does not prevent all indirect function calls
   for a given type from being checked.
2. All direct references to a function whose type has this attribute will always reference the
   function definition rather than an entry in the CFI jump table.
3. When a pointer to a function with this attribute is implicitly cast to a pointer to a function
   without this attribute, the compiler will give a warning saying this attribute is discarded. This
   warning can be silenced with an explicit cast. Note an explicit cast just disables the warning, so
   direct references to a function with a `cfi_unchecked_callee` attribute will still reference the
   function definition rather than the CFI jump table.

```c
#define CFI_UNCHECKED_CALLEE __attribute__((cfi_unchecked_callee))

void no_cfi() CFI_UNCHECKED_CALLEE {}

void (*with_cfi)() = no_cfi;  // warning: implicit conversion discards `cfi_unchecked_callee` attribute.
                              // `with_cfi` also points to the actual definition of `no_cfi` rather than
                              // its jump table entry.

void invoke(void (CFI_UNCHECKED_CALLEE *func)()) {
  func();  // CFI will not instrument this indirect call.

  void (*func2)() = func;  // warning: implicit conversion discards `cfi_unchecked_callee` attribute.

  func2();  // CFI will instrument this indirect call. Users should be careful however because if this
            // references a function with type `cfi_unchecked_callee`, then the CFI check may incorrectly
            // fail because the reference will be to the function definition rather than the CFI jump
            // table entry.
}
```

This attribute can only be applied on functions or member functions. This attribute can be a good
alternative to `no_sanitize("cfi")` if you only want to disable innstrumentation for specific indirect
calls rather than applying `no_sanitize("cfi")` on the whole function containing indirect call. Note
that `cfi_unchecked_attribute` is a type attribute doesn't disable CFI instrumentation on a function
body.


### clang_arm_mve_strict_polymorphism

{clang-attr-syntaxes}`ArmMveStrictPolymorphismDocs`

This attribute is used in the implementation of the ACLE intrinsics for the Arm
MVE instruction set. It is used to define the vector types used by the MVE
intrinsics.

Its effect is to modify the behavior of a vector type with respect to function
overloading. If a candidate function for overload resolution has a parameter
type with this attribute, then the selection of that candidate function will be
disallowed if the actual argument can only be converted via a lax vector
conversion. The aim is to prevent spurious ambiguity in ARM MVE polymorphic
intrinsics.

```c++
void overloaded(uint16x8_t vector, uint16_t scalar);
void overloaded(int32x4_t vector, int32_t scalar);
uint16x8_t myVector;
uint16_t myScalar;

// myScalar is promoted to int32_t as a side effect of the addition,
// so if lax vector conversions are considered for myVector, then
// the two overloads are equally good (one argument conversion
// each). But if the vector has the __clang_arm_mve_strict_polymorphism
// attribute, only the uint16x8_t,uint16_t overload will match.
overloaded(myVector, myScalar + 1);
```

However, this attribute does not prohibit lax vector conversions in contexts
other than overloading.

```c++
uint16x8_t function();

// This is still permitted with lax vector conversion enabled, even
// if the vector types have __clang_arm_mve_strict_polymorphism
int32x4_t result = function();
```


### cmse_nonsecure_call

{clang-attr-syntaxes}`ArmCmseNSCallDocs`

This attribute declares a non-secure function type. When compiling for secure
state, a call to such a function would switch from secure to non-secure state.
All non-secure function calls must happen only through a function pointer, and
a non-secure function type should only be used as a base type of a pointer.
See [ARMv8-M Security Extensions: Requirements on Development
Tools - Engineering Specification Documentation](https://developer.arm.com/docs/ecm0359818/latest/) for more information.


### contained_type

{clang-attr-syntaxes}`HLSLContainedTypeDocs`

The `hlsl::contained_type` attribute specifies the type of the HLSL resource
represented by a member variable of type `__hlsl_resource_t`.

This attribute is only valid for resource handles, and is an implementation
detail of clang's HLSL implementation. For more information see
{doc}`HLSL Resource Types <HLSL/ResourceTypes>`.


### device_builtin_surface_type

{clang-attr-syntaxes}`CUDADeviceBuiltinSurfaceTypeDocs`

The `device_builtin_surface_type` attribute can be applied to a class
template when declaring the surface reference. A surface reference variable
could be accessed on the host side and, on the device side, might be translated
into an internal surface object, which is established through surface bind and
unbind runtime APIs.


### device_builtin_texture_type

{clang-attr-syntaxes}`CUDADeviceBuiltinTextureTypeDocs`

The `device_builtin_texture_type` attribute can be applied to a class
template when declaring the texture reference. A texture reference variable
could be accessed on the host side and, on the device side, might be translated
into an internal texture object, which is established through texture bind and
unbind runtime APIs.


### dimension

{clang-attr-syntaxes}`HLSLResourceDimensionDocs`

The `hlsl::dimension` attribute specifies the dimensions of the HLSL resource
represented by a member variable of type `__hlsl_resource_t`, declaring the
resource to have Unknown, 1D, 2D, 3D, or Cube dimension.

This attribute is only valid for resource handles, and is an implementation
detail of clang's HLSL implementation. For more information see
{doc}`HLSL Resource Types <HLSL/ResourceTypes>`.


### enforce_read_only_placement

{clang-attr-syntaxes}`ReadOnlyPlacementDocs`

This attribute is attached to a structure, class or union declaration.

: When attached to a record declaration/definition, it checks if all instances
  of this type can be placed in the read-only data segment of the program. If it
  finds an instance that can not be placed in a read-only segment, the compiler
  emits a warning at the source location where the type was used.

  Examples:
  - `struct __attribute__((enforce_read_only_placement)) Foo;`
  - `struct __attribute__((enforce_read_only_placement)) Bar { ... };`

  Both `Foo` and `Bar` types have the `enforce_read_only_placement` attribute.

  The goal of introducing this attribute is to assist developers with writing secure
  code. A `const`-qualified global is generally placed in the read-only section
  of the memory that has additional run time protection from malicious writes. By
  attaching this attribute to a declaration, the developer can express the intent
  to place all instances of the annotated type in the read-only program memory.

  Note 1: The attribute doesn't guarantee that the object will be placed in the
  read-only data segment as it does not instruct the compiler to ensure such
  a placement. It emits a warning if something in the code can be proven to prevent
  an instance from being placed in the read-only data segment.

  Note 2: Currently, clang only checks if all global declarations of a given type `T`
  are `const`-qualified. The following conditions would also prevent the data to be
  put into read only segment, but the corresponding warnings are not yet implemented.

  1. An instance of type `T` is allocated on the heap/stack.
  2. Type `T` defines/inherits a mutable field.
  3. Type `T` defines/inherits non-constexpr constructor(s) for initialization.
  4. A field of type `T` is defined by type `Q`, which does not bear the
     `enforce_read_only_placement` attribute.
  5. A type `Q` inherits from type `T` and it does not have the
     `enforce_read_only_placement` attribute.


### is_array

{clang-attr-syntaxes}`HLSLIsArrayDocs`

The `hlsl::is_array` attribute specifies that the HLSL resource represented
by a member variable of type `__hlsl_resource_t` has array dimensions.

This attribute is only valid for resource handles, and is an implementation
detail of clang's HLSL implementation. For more information see
{doc}`HLSL Resource Types <HLSL/ResourceTypes>`.


### is_counter

{clang-attr-syntaxes}`HLSLIsCounterDocs`

The `hlsl::is_counter` attribute specifies that the HLSL resource represented
by a member variable of type `__hlsl_resource_t` is a counter buffer.

This attribute is only valid for resource handles, and is an implementation
detail of clang's HLSL implementation. For more information see
{doc}`HLSL Resource Types <HLSL/ResourceTypes>`.


### is_ms

{clang-attr-syntaxes}`HLSLIsMultiSampledDocs`

The `hlsl::is_array` attribute specifies that the HLSL resource represented
by a member variable of type `__hlsl_resource_t` is multisampled.

This attribute is only valid for resource handles, and is an implementation
detail of clang's HLSL implementation. For more information see
{doc}`HLSL Resource Types <HLSL/ResourceTypes>`.


### is_rov

{clang-attr-syntaxes}`HLSLIsROVDocs`

The `hlsl::is_rov` attribute specifies that the HLSL resource represented by
a member variable of type `__hlsl_resource_t` is a rasterizer ordered view.

This attribute is only valid for resource handles, and is an implementation
detail of clang's HLSL implementation. For more information see
{doc}`HLSL Resource Types <HLSL/ResourceTypes>`.


### noderef

{clang-attr-syntaxes}`NoDerefDocs`

The `noderef` attribute causes clang to diagnose dereferences of annotated pointer types.
This is ideally used with pointers that point to special memory which cannot be read
from or written to, but allowing for the pointer to be used in pointer arithmetic.
The following are examples of valid expressions where dereferences are diagnosed:

```c
int __attribute__((noderef)) *p;
int x = *p;  // warning

int __attribute__((noderef)) **p2;
x = **p2;  // warning

int * __attribute__((noderef)) *p3;
p = *p3;  // warning

struct S {
  int a;
};
struct S __attribute__((noderef)) *s;
x = s->a;    // warning
x = (*s).a;  // warning
```

Not all dereferences may diagnose a warning if the value directed by the pointer may not be
accessed. The following are examples of valid expressions where may not be diagnosed:

```c
int *q;
int __attribute__((noderef)) *p;
q = &*p;
q = *&p;

struct S {
  int a;
};
struct S __attribute__((noderef)) *s;
p = &s->a;
p = &(*s).a;
```

`noderef` is currently only supported for pointers and arrays and not usable
for references or Objective-C object pointers.

```c++
int x = 2;
int __attribute__((noderef)) &y = x;  // warning: 'noderef' can only be used on an array or pointer type
```

```objc
id __attribute__((noderef)) obj = [NSObject new]; // warning: 'noderef' can only be used on an array or pointer type
```


### objc_class_stub

{clang-attr-syntaxes}`ObjCClassStubDocs`

This attribute specifies that the Objective-C class to which it applies is
instantiated at runtime.

Unlike `__attribute__((objc_runtime_visible))`, a class having this attribute
still has a "class stub" that is visible to the linker. This allows categories
to be defined. Static message sends with the class as a receiver use a special
access pattern to ensure the class is lazily instantiated from the class stub.

Classes annotated with this attribute cannot be subclassed and cannot have
implementations defined for them. This attribute is intended for use in
Swift-generated headers for classes defined in Swift.

Adding or removing this attribute to a class is an ABI-breaking change.


### overflow_behavior

{clang-attr-syntaxes}`OverflowBehaviorDocs`

The `overflow_behavior` attribute provides fine-grained, type-level control
over how arithmetic operations on an integer type behave on overflow. It may be
applied to a `typedef`, to a variable or data member, or to an integer type
directly, and accepts one of two behaviors as its argument:

- `wrap`: arithmetic on the attributed type wraps on overflow, using two's
  complement semantics. This is equivalent to `-fwrapv` but scoped to the
  attributed type, and works for both signed and unsigned types. UBSan's
  `signed-integer-overflow`, `unsigned-integer-overflow`,
  `implicit-signed-integer-truncation`, and
  `implicit-unsigned-integer-truncation` checks are suppressed for the type.
- `trap`: arithmetic on the attributed type is checked for overflow, enabling
  overflow checks for the type even when `-fwrapv` is in effect globally.

```c++
typedef unsigned int __attribute__((overflow_behavior(trap))) non_wrapping_uint;

non_wrapping_uint add_one(non_wrapping_uint a) {
  return a + 1; // Overflow is checked for this operation.
}

int mul_alot(int n) {
  int __attribute__((overflow_behavior(wrap))) a = n;
  return a * 1337; // Overflow is not checked and is well-defined.
}
```

The keyword spellings `__ob_wrap` and `__ob_trap` are equivalent to
`overflow_behavior(wrap)` and `overflow_behavior(trap)` respectively.

The attribute wholly overrides global flags (`-ftrapv`, `-fwrapv`,
sanitizers, and Sanitizer Special Case Lists) for the attributed type. It can
only be applied to integer types.

This feature is experimental and must be enabled with the `-cc1` option
`-fexperimental-overflow-behavior-types`. For full details on promotion and
conversion rules, pointer semantics, diagnostics, and interaction with
sanitizers, see {doc}`OverflowBehaviorTypes`.


### raw_buffer

{clang-attr-syntaxes}`HLSLRawBufferDocs`

The `hlsl::raw_buffer` attribute specifies that the HLSL resource represented
by a member variable of type `__hlsl_resource_t` has raw buffer semantics.

This attribute is only valid for resource handles, and is an implementation
detail of clang's HLSL implementation. For more information see
{doc}`HLSL Resource Types <HLSL/ResourceTypes>`.


### resource_class

{clang-attr-syntaxes}`HLSLResourceClassDocs`

The `hlsl::resource_class` attribute specifies the resource class of the HLSL
resource represented by a member variable of type `__hlsl_resource_t`,
declaring it to be an SRV, UAV, CBuffer, or Sampler resource.

This attribute is only valid for resource handles, and is an implementation
detail of clang's HLSL implementation. For more information see
{doc}`HLSL Resource Types <HLSL/ResourceTypes>`.


### riscv_rvv_vector_bits

{clang-attr-syntaxes}`RISCVRVVVectorBitsDocs`

On RISC-V targets, the `riscv_rvv_vector_bits(N)` attribute is used to define
fixed-length variants of sizeless types.

For example:

```c
#include <riscv_vector.h>

#if defined(__riscv_v_fixed_vlen)
typedef vint8m1_t fixed_vint8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
#endif
```

Creates a type `fixed_vint8m1_t` that is a fixed-length variant of
`vint8m1_t` that contains exactly 512 bits. Unlike `vint8m1_t`, this type
can be used in globals, structs, unions, and arrays, all of which are
unsupported for sizeless types.

The attribute can be attached to a single RVV vector (such as `vint8m1_t`).
The attribute will be rejected unless
`N==(__riscv_v_fixed_vlen*LMUL)`, the implementation defined feature macro that
is enabled under the `-mrvv-vector-bits` flag. `__riscv_v_fixed_vlen` can
only be a power of 2 between 64 and 65536.

For types where LMUL!=1, `__riscv_v_fixed_vlen` needs to be scaled by the LMUL
of the type before passing to the attribute.

For `vbool*_t` types, `__riscv_v_fixed_vlen` needs to be divided by the
number from the type name. For example, `vbool8_t` needs to use
`__riscv_v_fixed_vlen` / 8. If the resulting value is not a multiple of 8,
the type is not supported for that value of `__riscv_v_fixed_vlen`.


### type_visibility

{clang-attr-syntaxes}`TypeVisibilityDocs`

The `type_visibility` attribute allows the visibility of a type and its vague
linkage objects (vtable, typeinfo, typeinfo name) to be controlled separately from
the visibility of functions and data members of the type.

For example, this can be used to give default visibility to the typeinfo and the vtable
of a type while still keeping hidden visibility on its member functions and static data
members.

This attribute can only be applied to types and namespaces.

If both `visibility` and `type_visibility` are applied to a type or a namespace, the
visibility specified with the `type_visibility` attribute overrides the visibility
provided with the regular `visibility` attribute.


### warn_unused

{clang-attr-syntaxes}`WarnUnusedDocs`

The `warn_unused` attribute can be placed on the declaration of a structure or union type.
When the `-Wunused-variable` diagnostic is enabled, local variables of types which have a non-trivial constructor or destructor are considered "used" by virtue of the constructor or destructor invocations involved.
Those constructor or destructor invocations are not considered a use if the type is declared with the `warn_unused` attribute.
The variable is considered used if it is named outside of its declaration.

This attribute is available in both C and C++ language modes but is primarily useful in C++ for classes which have a non-trivial constructor or destructor but act as a value type rather than an RAII type.

```c++
struct [[gnu::warn_unused]] S {
  S();
  ~S();
};

struct T {
  T();
  ~T();
 };

 int func() {
   S s1; // -Wunused-variable warning
   S s2; // No -Wunused-variable warning because of the member access expression below
   S s3; // No -Wunused-variable warning because of the sizeof operand below
   T t;  // No -Wunused-variable warning

   s2.~S();
   return sizeof(s3);
 }
```


