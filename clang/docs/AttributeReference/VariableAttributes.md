## Variable Attributes



### HLSL Interpolation Modifiers

{clang-attr-syntaxes}`HLSLInterpolationModifierDocs`

The HLSL keywords `nointerpolation`, `linear`, `centroid`,
`noperspective`, `sample`, and `center` control interpolation of pixel
shader inputs and vertex shader outputs.

When applied to an aggregate type, the modifier propagates recursively to every
scalar or vector of the type. A modifier on an inner field overrides one
inherited from an enclosing declaration.

`nointerpolation` cannot be combined with another interpolation modifier.
For pixel shader inputs and vertex shader outputs, it cannot be used on
`SV_Position`. Integer, boolean, and 64-bit floating-point components only
support `nointerpolation`.

Unqualified pixel shader inputs and vertex shader outputs default to `linear`
for floating-point components of at most 32 bits and `nointerpolation` otherwise.
`SV_Position` uses the corresponding `noperspective` mode. Vertex shader inputs,
pixel shader outputs, and signatures in other shader stages have no
interpolation mode; their interpolation modifiers are ignored.


### HLSL Parameter Modifiers

{clang-attr-syntaxes}`HLSLParamQualifierDocs`

HLSL function parameters are passed by value. Parameter declarations support
three qualifiers to denote parameter passing behavior. The three qualifiers are
`in`, `out` and `inout`.

Parameters annotated with `in` or with no annotation are passed by value from
the caller to the callee.

Parameters annotated with `out` are written to the argument after the callee
returns (Note: arguments values passed into `out` parameters *are not* copied
into the callee).

Parameters annotated with `inout` are copied into the callee via a temporary,
and copied back to the argument after the callee returns.


### __ptrauth

{clang-attr-syntaxes}`PtrAuthDocs`

The `__ptrauth` qualifier allows the programmer to directly control
how pointers are signed when they are stored in a particular variable.
This can be used to strengthen the default protections of pointer
authentication and make it more difficult for an attacker to escalate
an ability to alter memory into full control of a process.

```c
#include <ptrauth.h>

typedef void (*my_callback)(const void*);
my_callback __ptrauth(ptrauth_key_process_dependent_code, 1, 0xe27a) callback;
```

The first argument to `__ptrauth` is the name of the signing key.
Valid key names for the target are defined in `<ptrauth.h>`.

The second argument to `__ptrauth` is a flag (0 or 1) specifying whether
the object should use address discrimination.

The third argument to `__ptrauth` is a 16-bit non-negative integer which
allows additional discrimination between objects.


### always_destroy

{clang-attr-syntaxes}`AlwaysDestroyDocs`

The `always_destroy` attribute specifies that a variable with static or thread
storage duration should have its exit-time destructor run. This attribute is the
default unless clang was invoked with `-fno-c++-static-destructors`.

If a variable is explicitly declared with this attribute, Clang will silence
otherwise applicable `-Wexit-time-destructors` warnings.


### binding

{clang-attr-syntaxes}`HLSLVkBindingDocs`

The `[[vk::binding]]` attribute allows you to explicitly specify the descriptor
set and binding for a resource when targeting SPIR-V. This is particularly
useful when you need different bindings for SPIR-V and DXIL, as the `register`
attribute can be used for DXIL-specific bindings.

The attribute takes two integer arguments: the binding and the descriptor set.
The descriptor set is optional and defaults to 0 if not provided.

```c++
// A structured buffer with binding 23 in descriptor set 102.
[[vk::binding(23, 102)]] StructuredBuffer<float> Buf;

// A structured buffer with binding 14 in descriptor set 0.
[[vk::binding(14)]] StructuredBuffer<float> Buf2;

// A cbuffer with binding 1 in descriptor set 2.
[[vk::binding(1, 2)]] cbuffer MyCBuffer {
  float4x4 worldViewProj;
};
```


### called_once

{clang-attr-syntaxes}`CalledOnceDocs`

The `called_once` attribute specifies that the annotated function or method
parameter is invoked exactly once on all execution paths. It only applies
to parameters with function-like types, i.e. function pointers or blocks. This
concept is particularly useful for asynchronous programs.

Clang implements a check for `called_once` parameters,
`-Wcalled-once-parameter`. It is on by default and finds the following
violations:

- Parameter is not called at all.
- Parameter is called more than once.
- Parameter is not called on one of the execution paths.

In the latter case, Clang pinpoints the path where parameter is not invoked
by showing the control-flow statement where the path diverges.

```objc
void fooWithCallback(void (^callback)(void) __attribute__((called_once))) {
  if (somePredicate()) {
    ...
    callback();
  } else {
    callback(); // OK: callback is called on every path
  }
}

void barWithCallback(void (^callback)(void) __attribute__((called_once))) {
  if (somePredicate()) {
    ...
    callback(); // note: previous call is here
  }
  callback(); // warning: callback is called twice
}

void foobarWithCallback(void (^callback)(void) __attribute__((called_once))) {
  if (somePredicate()) {  // warning: callback is not called when condition is false
    ...
    callback();
  }
}
```

This attribute is useful for API developers who want to double-check if they
implemented their method correctly.


### clang::code_align

{clang-attr-syntaxes}`CodeAlignAttrDocs`

The `clang::code_align(N)` attribute applies to a loop and specifies the byte
alignment for a loop. The attribute accepts a positive integer constant
initialization expression indicating the number of bytes for the minimum
alignment boundary. Its value must be a power of 2, between 1 and 4096
(inclusive).

```c++
void foo() {
  int var = 0;
  [[clang::code_align(16)]] for (int i = 0; i < 10; ++i) var++;
}

void Array(int *array, size_t n) {
  [[clang::code_align(64)]] for (int i = 0; i < n; ++i) array[i] = 0;
}

void count () {
  int a1[10], int i = 0;
  [[clang::code_align(32)]] while (i < 10) { a1[i] += 3; }
}

void check() {
  int a = 10;
  [[clang::code_align(8)]] do {
    a = a + 1;
  } while (a < 20);
}

template<int A>
void func() {
  [[clang::code_align(A)]] for(;;) { }
}
```


### cleanup

{clang-attr-syntaxes}`CleanupDocs`

This attribute allows a function to be run when a local variable goes out of
scope. The attribute takes the identifier of a function with a parameter type
that is a pointer to the type with the attribute.

```c
static void foo (int *) { ... }
static void bar (int *) { ... }
void baz (void) {
  int x __attribute__((cleanup(foo)));
  {
    int y __attribute__((cleanup(bar)));
  }
}
```

The above example will result in a call to `bar` being passed the address of
`y` when `y` goes out of scope, then a call to `foo` being passed the
address of `x` when `x` goes out of scope. If two or more variables share
the same scope, their `cleanup` callbacks are invoked in the reverse order
the variables were declared in. It is not possible to check the return value
(if any) of these `cleanup` callback functions.


### dllexport

{clang-attr-syntaxes}`DLLExportDocs`

The `__declspec(dllexport)` attribute declares a variable, function, or
Objective-C interface to be exported from the module. It is available under the
`-fdeclspec` flag for compatibility with various compilers. The primary use
is for COFF object files which explicitly specify what interfaces are available
for external use. See the [dllexport][dllexport] documentation on MSDN for more
information.

[dllexport]: https://msdn.microsoft.com/en-us/library/3y1sfaz2.aspx


### dllimport

{clang-attr-syntaxes}`DLLImportDocs`

The `__declspec(dllimport)` attribute declares a variable, function, or
Objective-C interface to be imported from an external module. It is available
under the `-fdeclspec` flag for compatibility with various compilers. The
primary use is for COFF object files which explicitly specify what interfaces
are imported from external modules. See the [dllimport][dllimport] documentation on MSDN
for more information.

Note that a dllimport function may still be inlined, if its definition is
available and it doesn't reference any non-dllimport functions or global
variables.

[dllimport]: https://msdn.microsoft.com/en-us/library/3y1sfaz2.aspx


### ext_builtin_input

{clang-attr-syntaxes}`HLSLVkExtBuiltinInputDocs`

Vulkan shaders have `Input` builtins. Those variables are externally
initialized by the driver/pipeline, but each copy is private to the current
lane.

Those builtins can be declared using the `[[vk::ext_builtin_input]]` attribute
like follows:

```c++
[[vk::ext_builtin_input(/* WorkgroupId */ 26)]]
static const uint3 groupid;
```

This variable will be lowered into a module-level variable, with the `Input`
storage class, and the `BuiltIn 26` decoration.

The full documentation for this inline SPIR-V attribute can be found here:
<https://github.com/microsoft/hlsl-specs/blob/main/proposals/0011-inline-spirv.md>


### ext_builtin_output

{clang-attr-syntaxes}`HLSLVkExtBuiltinOutputDocs`

Vulkan shaders have `Output` builtins. Those variables are externally
visible to the driver/pipeline, but each copy is private to the current
lane.

Those builtins can be declared using the `[[vk::ext_builtin_output]]`
attribute like follows:

```c++
[[vk::ext_builtin_output(/* Position */ 0)]]
static float4 position;
```

This variable will be lowered into a module-level variable, with the `Output`
storage class, and the `BuiltIn 0` decoration.

The full documentation for this inline SPIR-V attribute can be found here:
<https://github.com/microsoft/hlsl-specs/blob/main/proposals/0011-inline-spirv.md>


### groupshared

{clang-attr-syntaxes}`HLSLGroupSharedAddressSpaceDocs`

HLSL enables threads of a compute shader to exchange values via shared memory.
HLSL provides barrier primitives such as GroupMemoryBarrierWithGroupSync,
and so on to ensure the correct ordering of reads and writes to shared memory
in the shader and to avoid data races.
Here's an example to declare a groupshared variable.

```c++
groupshared GSData data[5*5*1];
```

The full documentation is available here: <https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-variable-syntax#group-shared>


### init_priority

{clang-attr-syntaxes}`InitPriorityDocs`

In C++, the order in which global variables are initialized across translation
units is unspecified, unlike the ordering within a single translation unit. The
`init_priority` attribute allows you to specify a relative ordering for the
initialization of objects declared at namespace scope in C++ within a single
linked image on supported platforms. The priority is given as an integer constant
expression between 101 and 65535 (inclusive). Priorities outside of that range are
reserved for use by the implementation. A lower value indicates a higher priority
of initialization. Note that only the relative ordering of values is important.
For example:

```c++
struct SomeType { SomeType(); };
__attribute__((init_priority(200))) SomeType Obj1;
__attribute__((init_priority(101))) SomeType Obj2;
```

`Obj2` will be initialized *before* `Obj1` despite the usual order of
initialization being the opposite.

Note that this attribute does not control the initialization order of objects
across final linked image boundaries like shared objects and executables.

On Windows, `init_seg(compiler)` is represented with a priority of 200 and
`init_seg(library)` is represented with a priority of 400. `init_seg(user)`
uses the default 65535 priority.

On MachO platforms, this attribute also does not control the order of initialization
across translation units, where it only affects the order within a single TU.

This attribute is only supported for C++ and Objective-C++ and is ignored in
other language modes.


### init_seg

{clang-attr-syntaxes}`InitSegDocs`

The attribute applied by `pragma init_seg()` controls the section into
which global initialization function pointers are emitted. It is only
available with `-fms-extensions`. Typically, this function pointer is
emitted into `.CRT$XCU` on Windows. The user can change the order of
initialization by using a different section name with the same
`.CRT$XC` prefix and a suffix that sorts lexicographically before or
after the standard `.CRT$XCU` sections. See the [init_seg][init_seg]
documentation on MSDN for more information.

[init_seg]: http://msdn.microsoft.com/en-us/library/7977wcck(v=vs.110).aspx


### leaf

{clang-attr-syntaxes}`LeafDocs`

The `leaf` attribute is used as a compiler hint to improve dataflow analysis
in library functions. Functions marked with the `leaf` attribute are not allowed
to jump back into the caller's translation unit, whether through invoking a
callback function, an external function call, use of `longjmp`, or other means.
Therefore, they cannot use or modify any data that does not escape the caller function's
compilation unit.

For more information see the
[GCC common attributes documentation](https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html)


### loader_uninitialized

{clang-attr-syntaxes}`LoaderUninitializedDocs`

The `loader_uninitialized` attribute can be placed on global variables to
indicate that the variable does not need to be zero initialized by the loader.
On most targets, zero-initialization does not incur any additional cost.
For example, most general purpose operating systems deliberately ensure
that all memory is properly initialized in order to avoid leaking privileged
information from the kernel or other programs. However, some targets
do not make this guarantee, and on these targets, avoiding an unnecessary
zero-initialization can have a significant impact on load times and/or code
size.

A declaration with this attribute is a non-tentative definition just as if it
provided an initializer. Variables with this attribute are considered to be
uninitialized in the same sense as a local variable, and the programs must
write to them before reading from them. If the variable's type is a C++ class
type with a non-trivial default constructor, or an array thereof, this attribute
only suppresses the static zero-initialization of the variable, not the dynamic
initialization provided by executing the default constructor.


### location

{clang-attr-syntaxes}`HLSLVkLocationDocs`

Attribute used for specifying the location number for the stage input/output
variables. Allowed on function parameters, function returns, and struct
fields. This parameter has no effect when used outside of an entrypoint
parameter/parameter field/return value.

This attribute maps to the `Location` SPIR-V decoration.


### maybe_undef

{clang-attr-syntaxes}`MaybeUndefDocs`

The `maybe_undef` attribute can be placed on a function parameter. It indicates
that the parameter is allowed to use undef values. It informs the compiler
to insert a freeze LLVM IR instruction on the function parameter.
Please note that this is an attribute that is used as an internal
implementation detail and not intended to be used by external users.

In languages HIP, CUDA etc., some functions have multi-threaded semantics and
it is enough for only one or some threads to provide defined arguments.
Depending on semantics, undef arguments in some threads don't produce
undefined results in the function call. Since, these functions accept undefined
arguments, `maybe_undef` attribute can be placed.

Sample usage:

```c
void maybeundeffunc(int __attribute__((maybe_undef))param);
```


### maybe_unused, unused

{clang-attr-syntaxes}`WarnMaybeUnusedDocs`

When passing the `-Wunused` flag to Clang, entities that are unused by the
program may be diagnosed. The `[[maybe_unused]]` (or
`__attribute__((unused))`) attribute can be used to silence such diagnostics
when the entity cannot be removed. For instance, a local variable may exist
solely for use in an `assert()` statement, which makes the local variable
unused when `NDEBUG` is defined.

The attribute may be applied to the declaration of a class, a typedef, a
variable, a function or method, a function parameter, an enumeration, an
enumerator, a non-static data member, or a label.

```c++
#include <cassert>

[[maybe_unused]] void f([[maybe_unused]] bool thing1,
                        [[maybe_unused]] bool thing2) {
  [[maybe_unused]] bool b = thing1 && thing2;
  assert(b);
}
```


### model

{clang-attr-syntaxes}`CodeModelDocs`

The `model` attribute allows overriding the translation unit's
code model (specified by `-mcmodel`) for a specific global variable.

On LoongArch, allowed values are "normal", "medium", "extreme".

On x86-64, allowed values are `"small"` and `"large"`. `"small"` is
roughly equivalent to `-mcmodel=small`, meaning the global is considered
"small" placed closer to the `.text` section relative to "large" globals, and
to prefer using 32-bit relocations to access the global. `"large"` is roughly
equivalent to `-mcmodel=large`, meaning the global is considered "large" and
placed further from the `.text` section relative to "small" globals, and
64-bit relocations must be used to access the global.


### no_destroy

{clang-attr-syntaxes}`NoDestroyDocs`

The `no_destroy` attribute specifies that a variable with static or thread
storage duration shouldn't have its exit-time destructor run. Annotating every
static and thread duration variable with this attribute is equivalent to
invoking clang with `-fno-c++-static-destructors`.

If a variable is declared with this attribute, clang doesn't access check or
generate the type's destructor. If you have a type that you only want to be
annotated with `no_destroy`, you can therefore declare the destructor private:

```c++
struct only_no_destroy {
  only_no_destroy();
private:
  ~only_no_destroy();
};

[[clang::no_destroy]] only_no_destroy global; // fine!
```

Note that destructors are still required for subobjects of aggregates annotated
with this attribute. This is because previously constructed subobjects need to
be destroyed if an exception gets thrown before the initialization of the
complete object is complete. For instance:

```c++
void f() {
  try {
    [[clang::no_destroy]]
    static only_no_destroy array[10]; // error, only_no_destroy has a private destructor.
  } catch (...) {
    // Handle the error
  }
}
```

Here, if the construction of `array[9]` fails with an exception, `array[0..8]`
will be destroyed, so the element's destructor needs to be accessible.


### nodebug

{clang-attr-syntaxes}`NoDebugDocs`

The `nodebug` attribute allows you to suppress debugging information for a
function or method, for a variable that is not a parameter or a non-static
data member, or for a typedef or using declaration.


### noescape

{clang-attr-syntaxes}`NoEscapeDocs`

`noescape` placed on a function parameter of a pointer type is used to inform
the compiler that the pointer cannot escape: that is, no reference to the object
the pointer points to that is derived from the parameter value will survive
after the function returns. Users are responsible for making sure parameters
annotated with `noescape` do not actually escape. The optimizer may make
assumptions based on the fact that it knows that a call to the function does
not escape a certain parameter, so incorrectly annotating a parameter with
`noescape` leads to undefined behavior. The callee is also not allowed to
deallocate memory through a `noescape` parameter: the optimizer does not make
assumptions based on this information at the moment, but may do so in the
future. Some cases of invalid uses of `noescape` can be found with
{ref}`-Wlifetime-safety-noescape <Wlifetime-safety-noescape>`.

For example:

```c
int *gp;

void nonescapingFunc(__attribute__((noescape)) int *p) {
  *p += 100; // OK.
}

void escapingFunc(__attribute__((noescape)) int *p) {
  gp = p; // Not OK.
}

void freeingFunc(__attribute__((noescape)) int *p) {
  free(p); // Not OK.
}
```

Since `noescape` is a parameter attribute and not a type attribute, it only
applies to the outermost pointer level, regardless of where in the parameter
declaration you place it:

```c
int **gp;

void nestingEscapes(__attribute__((noescape)) int **p) {
  gp = p; // Not OK.
  *gp = *p; // OK, p does not escape.
}
```

Additionally, when the parameter is a
{doc}`block pointer <BlockLanguageSpec>`, the same restriction applies to
copies of the block. For example:

```c
typedef void (^BlockTy)();
BlockTy g0, g1;

void nonescapingFunc(__attribute__((noescape)) BlockTy block) {
  block(); // OK.
}

void escapingFunc(__attribute__((noescape)) BlockTy block) {
  g0 = block; // Not OK.
  g1 = Block_copy(block); // Not OK either.
}
```

The function *is* allowed to leak information about the memory address of the
pointer, but not any provenance of the allocation:

```c
bool isNull(__attribute__((noescape)) void *p) {
  return !p; // OK.
}

uintptr_t gi;

void escapingAddress(__attribute__((noescape)) int *p) {
  // OK *if and only if* gi is never casted back to a pointer.
  gi = (uintptr_t)p;
}

bool usingEscapedAddress(int *p) {
  return (uintptr_t)p > gi; // OK.
}

bool usingEscapedPointer(int *p) {
  return p > (int*)gi; // Not OK.
}

int *gp;

void escapingEndFunc(__attribute__((noescape)) int *p, size_t len) {
  gp = p + len; // Not OK.
}
```


### nosvm

{clang-attr-syntaxes}`OpenCLNoSVMDocs`

OpenCL 2.0 supports the optional `__attribute__((nosvm))` qualifier for
pointer variable. It informs the compiler that the pointer does not refer
to a shared virtual memory region. See OpenCL v2.0 s6.7.2 for details.

Since it is not widely used and has been removed from OpenCL 2.1, it is ignored
by Clang.


### objc_externally_retained

{clang-attr-syntaxes}`ObjCExternallyRetainedDocs`

The `objc_externally_retained` attribute can be applied to strong local
variables, functions, methods, or blocks to opt into
{ref}`externally-retained semantics <arc.misc.externally_retained>`.

When applied to the definition of a function, method, or block, every parameter
of the function with implicit strong retainable object pointer type is
considered externally-retained, and becomes `const`. By explicitly annotating
a parameter with `__strong`, you can opt back into the default
non-externally-retained behavior for that parameter. For instance,
`first_param` is externally-retained below, but not `second_param`:

```objc
__attribute__((objc_externally_retained))
void f(NSArray *first_param, __strong NSArray *second_param) {
  // ...
}
```

Likewise, when applied to a strong local variable, that variable becomes
`const` and is considered externally-retained.

When compiled without `-fobjc-arc`, this attribute is ignored.


### pass_object_size, pass_dynamic_object_size

{clang-attr-syntaxes}`PassObjectSizeDocs`

:::{Note}
The mangling of functions with parameters that are annotated with
`pass_object_size` is subject to change. You can get around this by
using `__asm__("foo")` to explicitly name your functions, thus preserving
your ABI; also, non-overloadable C functions with `pass_object_size` are
not mangled.
:::

The `pass_object_size(Type)` attribute can be placed on function parameters to
instruct clang to call `__builtin_object_size(param, Type)` at each callsite
of said function, and implicitly pass the result of this call in as an invisible
argument of type `size_t` directly after the parameter annotated with
`pass_object_size`. Clang will also replace any calls to
`__builtin_object_size(param, Type)` in the function by said implicit
parameter.

Example usage:

```c
int bzero1(char *const p __attribute__((pass_object_size(0))))
    __attribute__((noinline)) {
  int i = 0;
  for (/**/; i < (int)__builtin_object_size(p, 0); ++i) {
    p[i] = 0;
  }
  return i;
}

int main() {
  char chars[100];
  int n = bzero1(&chars[0]);
  assert(n == sizeof(chars));
  return 0;
}
```

If successfully evaluating `__builtin_object_size(param, Type)` at the
callsite is not possible, then the "failed" value is passed in. So, using the
definition of `bzero1` from above, the following code would exit cleanly:

```c
int main2(int argc, char *argv[]) {
  int n = bzero1(argv);
  assert(n == -1);
  return 0;
}
```

`pass_object_size` plays a part in overload resolution. If two overload
candidates are otherwise equally good, then the overload with one or more
parameters with `pass_object_size` is preferred. This implies that the choice
between two identical overloads both with `pass_object_size` on one or more
parameters will always be ambiguous; for this reason, having two such overloads
is illegal. For example:

```c++
#define PS(N) __attribute__((pass_object_size(N)))
// OK
void Foo(char *a, char *b); // Overload A
// OK -- overload A has no parameters with pass_object_size.
void Foo(char *a PS(0), char *b PS(0)); // Overload B
// Error -- Same signature (sans pass_object_size) as overload B, and both
// overloads have one or more parameters with the pass_object_size attribute.
void Foo(void *a PS(0), void *b);

// OK
void Bar(void *a PS(0)); // Overload C
// OK
void Bar(char *c PS(1)); // Overload D

void main() {
  char known[10], *unknown;
  Foo(unknown, unknown); // Calls overload B
  Foo(known, unknown); // Calls overload B
  Foo(unknown, known); // Calls overload B
  Foo(known, known); // Calls overload B

  Bar(known); // Calls overload D
  Bar(unknown); // Calls overload D
}
```

Currently, `pass_object_size` is a bit restricted in terms of its usage:

- Only one use of `pass_object_size` is allowed per parameter.
- It is an error to take the address of a function with `pass_object_size` on
  any of its parameters. If you wish to do this, you can create an overload
  without `pass_object_size` on any parameters.
- It is an error to apply the `pass_object_size` attribute to parameters that
  are not pointers. Additionally, any parameter that `pass_object_size` is
  applied to must be marked `const` at its function's definition.

Clang also supports the `pass_dynamic_object_size` attribute, which behaves
identically to `pass_object_size`, but evaluates a call to
`__builtin_dynamic_object_size` at the callee instead of
`__builtin_object_size`. `__builtin_dynamic_object_size` provides some extra
runtime checks when the object size can't be determined at compile-time. You can
read more about `__builtin_dynamic_object_size` in
{ref}`Evaluating Object Size <langext-evaluating-object-size>`.


### push_constant

{clang-attr-syntaxes}`HLSLVkPushConstantDocs`

Vulkan shaders have `PushConstants`

The `[[vk::push_constant]]` attribute allows you to declare this
global variable as a push constant when targeting Vulkan.
This attribute is ignored otherwise.

This attribute must be applied to the variable, not underlying type.
The variable type must be a struct, per the requirements of Vulkan, "there
must be no more than one push constant block statically used per shader entry
point."


### require_constant_initialization, constinit (C++20)

{clang-attr-syntaxes}`ConstInitDocs`

This attribute specifies that the variable to which it is attached is intended
to have a [*constant initializer*](http://en.cppreference.com/w/cpp/language/constant_initialization)
according to the rules of [basic.start.static]. The variable is required to
have static or thread storage duration. If the initialization of the variable
is not a constant initializer an error will be produced. This attribute may
only be used in C++; the `constinit` spelling is only accepted in C++20
onwards.

Note that in C++03 strict constant expression checking is not done. Instead
the attribute reports if Clang can emit the variable as a constant, even if it's
not technically a *constant initializer*. This behavior is non-portable.

Static storage duration variables with constant initializers avoid hard-to-find
bugs caused by the indeterminate order of dynamic initialization. They can also
be safely used during dynamic initialization across translation units.

This attribute acts as a compile time assertion that the requirements
for constant initialization have been met. Since these requirements change
between dialects and have subtle pitfalls it's important to fail fast instead
of silently falling back on dynamic initialization.

The first use of the attribute on a variable must be part of, or precede, the
initializing declaration of the variable. C++20 requires the `constinit`
spelling of the attribute to be present on the initializing declaration if it
is used anywhere. The other spellings can be specified on a forward declaration
and omitted on a later initializing declaration.

```c++
// -std=c++14
#define SAFE_STATIC [[clang::require_constant_initialization]]
struct T {
  constexpr T(int) {}
  ~T(); // non-trivial
};
SAFE_STATIC T x = {42}; // Initialization OK. Doesn't check destructor.
SAFE_STATIC T y = 42; // error: variable does not have a constant initializer
// copy initialization is not a constant expression on a non-literal type.
```


### row_major, column_major

{clang-attr-syntaxes}`HLSLMatrixLayoutDocs`

The `row_major` and `column_major` keywords specify the memory layout
of an HLSL matrix type.

- `row_major`: Matrices are stored in memory row-by-row.
- `column_major`: Matrices are stored in memory column-by-column (default).

Example:

```hlsl
row_major float2x2 myMatrix;
```


### section, __declspec(allocate)

{clang-attr-syntaxes}`SectionDocs`

The `section` attribute allows you to specify a specific section a
global variable or function should be in after translation.


### standalone_debug

{clang-attr-syntaxes}`StandaloneDebugDocs`

The `standalone_debug` attribute causes debug info to be emitted for a record
type regardless of the debug info optimizations that are enabled with
-fno-standalone-debug. This attribute only has an effect when debug info
optimizations are enabled (e.g. with -fno-standalone-debug), and is C++-only.


### swift_async_context

{clang-attr-syntaxes}`SwiftAsyncContextDocs`

The `swift_async_context` attribute marks a parameter of a `swiftasynccall`
function as having the special asynchronous context-parameter ABI treatment.

If the function is not `swiftasynccall`, this attribute only generates
extended frame information.

A context parameter must have pointer or reference type.


### swift_context

{clang-attr-syntaxes}`SwiftContextDocs`

The `swift_context` attribute marks a parameter of a `swiftcall`
or `swiftasynccall` function as having the special context-parameter
ABI treatment.

This treatment generally passes the context value in a special register
which is normally callee-preserved.

A `swift_context` parameter must either be the last parameter or must be
followed by a `swift_error_result` parameter (which itself must always be
the last parameter).

A context parameter must have pointer or reference type.


### swift_error_result

{clang-attr-syntaxes}`SwiftErrorResultDocs`

The `swift_error_result` attribute marks a parameter of a `swiftcall`
function as having the special error-result ABI treatment.

This treatment generally passes the underlying error value in and out of
the function through a special register which is normally callee-preserved.
This is modeled in C by pretending that the register is addressable memory:

- The caller appears to pass the address of a variable of pointer type.
  The current value of this variable is copied into the register before
  the call; if the call returns normally, the value is copied back into the
  variable.
- The callee appears to receive the address of a variable. This address
  is actually a hidden location in its own stack, initialized with the
  value of the register upon entry. When the function returns normally,
  the value in that hidden location is written back to the register.

A `swift_error_result` parameter must be the last parameter, and it must be
preceded by a `swift_context` parameter.

A `swift_error_result` parameter must have type `T**` or `T*&` for some
type T. Note that no qualifiers are permitted on the intermediate level.

It is undefined behavior if the caller does not pass a pointer or
reference to a valid object.

The standard convention is that the error value itself (that is, the
value stored in the apparent argument) will be null upon function entry,
but this is not enforced by the ABI.


### swift_indirect_result

{clang-attr-syntaxes}`SwiftIndirectResultDocs`

The `swift_indirect_result` attribute marks a parameter of a `swiftcall`
or `swiftasynccall` function as having the special indirect-result ABI
treatment.

This treatment gives the parameter the target's normal indirect-result
ABI treatment, which may involve passing it differently from an ordinary
parameter. However, only the first indirect result will receive this
treatment. Furthermore, low-level lowering may decide that a direct result
must be returned indirectly; if so, this will take priority over the
`swift_indirect_result` parameters.

A `swift_indirect_result` parameter must either be the first parameter or
follow another `swift_indirect_result` parameter.

A `swift_indirect_result` parameter must have type `T*` or `T&` for
some object type `T`. If `T` is a complete type at the point of
definition of a function, it is undefined behavior if the argument
value does not point to storage of adequate size and alignment for a
value of type `T`.

Making indirect results explicit in the signature allows C functions to
directly construct objects into them without relying on language
optimizations like C++'s named return value optimization (NRVO).


### swiftasynccall

{clang-attr-syntaxes}`SwiftAsyncCallDocs`

The `swiftasynccall` attribute indicates that a function is
compatible with the low-level conventions of Swift async functions,
provided it declares the right formal arguments.

In most respects, this is similar to the `swiftcall` attribute, except for
the following:

- A parameter may be marked `swift_async_context`, `swift_context`
  or `swift_indirect_result` (with the same restrictions on parameter
  ordering as `swiftcall`) but the parameter attribute
  `swift_error_result` is not permitted.
- A `swiftasynccall` function must have return type `void`.
- Within a `swiftasynccall` function, a call to a `swiftasynccall`
  function that is the immediate operand of a `return` statement is
  guaranteed to be performed as a tail call. This syntax is allowed even
  in C as an extension (a call to a void-returning function cannot be a
  return operand in standard C). If something in the calling function would
  semantically be performed after a guaranteed tail call, such as the
  non-trivial destruction of a local variable or temporary,
  then the program is ill-formed.

Query for this attribute with `__has_attribute(swiftasynccall)`. Query if
the target supports the calling convention with
`__has_extension(swiftasynccc)`.

Since this attribute follows the Swift async calling convention, it is
considered ABI-unstable except on targets where the Swift project
has declared ABI stability. Users are responsible for ensuring that
calls and definitions of functions with this attribute are compiled
with compatible compilers. Note that different operating systems
on the same architecture may use different ABIs and therefore may
have different standards for ABI stability.


### swiftcall

{clang-attr-syntaxes}`SwiftCallDocs`

The `swiftcall` attribute indicates that a function should be called
using the Swift calling convention for a function or function pointer.

The lowering for the Swift calling convention, as described by the Swift
ABI documentation, occurs in multiple phases. The first, "high-level"
phase breaks down the formal parameters and results into innately direct
and indirect components, adds implicit parameters for the generic
signature, and assigns the context and error ABI treatments to parameters
where applicable. The second phase breaks down the direct parameters
and results from the first phase and assigns them to registers or the
stack. The `swiftcall` convention only handles this second phase of
lowering; the C function type must accurately reflect the results
of the first phase, as follows:

- Results classified as indirect by high-level lowering should be
  represented as parameters with the `swift_indirect_result` attribute.

- Results classified as direct by high-level lowering should be represented
  as follows:

  - First, remove any empty direct results.
  - If there are no direct results, the C result type should be `void`.
  - If there is one direct result, the C result type should be a type with
    the exact layout of that result type.
  - If there are a multiple direct results, the C result type should be
    a struct type with the exact layout of a tuple of those results.

- Parameters classified as indirect by high-level lowering should be
  represented as parameters of pointer type.

- Parameters classified as direct by high-level lowering should be
  omitted if they are empty types; otherwise, they should be represented
  as a parameter type with a layout exactly matching the layout of the
  Swift parameter type.

- The context parameter, if present, should be represented as a trailing
  parameter with the `swift_context` attribute.

- The error result parameter, if present, should be represented as a
  trailing parameter (always following a context parameter) with the
  `swift_error_result` attribute.

`swiftcall` does not support variadic arguments or unprototyped functions.

The parameter ABI treatment attributes are aspects of the function type.
A function type which applies an ABI treatment attribute to a
parameter is a different type from an otherwise-identical function type
that does not. A single parameter may not have multiple ABI treatment
attributes.

Support for this feature is target-dependent, although it should be
supported on every target that Swift supports. Query for this attribute
with `__has_attribute(swiftcall)`. Query if the target supports the
calling convention with `__has_extension(swiftcc)`. This implies
support for the `swift_context`, `swift_error_result`, and
`swift_indirect_result` attributes.

Since this attribute follows the Swift calling convention, it is
considered ABI-unstable except on targets where the Swift project
has declared ABI stability. Users are responsible for ensuring that
calls and definitions of functions with this attribute are compiled
with compatible compilers. Note that different operating systems
on the same architecture may use different ABIs and therefore may
have different standards for ABI stability.


### thread

{clang-attr-syntaxes}`ThreadDocs`

The `__declspec(thread)` attribute declares a variable with thread local
storage. It is available under the `-fms-extensions` flag for MSVC
compatibility. See the documentation for [`__declspec(thread)`][__declspec(thread)] on MSDN.

In Clang, `__declspec(thread)` is generally equivalent in functionality to the
GNU `__thread` keyword. The variable must not have a destructor and must have
a constant initializer, if any. The attribute only applies to variables
declared with static storage duration, such as globals, class static data
members, and static locals.

[__declspec(thread)]: http://msdn.microsoft.com/en-us/library/9w1sdazb.aspx


### tls_model

{clang-attr-syntaxes}`TLSModelDocs`

The `tls_model` attribute allows you to specify which thread-local storage
model to use. It accepts the following strings:

- global-dynamic
- local-dynamic
- initial-exec
- local-exec

TLS models are mutually exclusive.


### uninitialized

{clang-attr-syntaxes}`UninitializedDocs`

The command-line parameter `-ftrivial-auto-var-init=*` can be used to
initialize trivial automatic stack variables. By default, trivial automatic
stack variables are uninitialized. This attribute is used to override the
command-line parameter, forcing variables to remain uninitialized. It has no
semantic meaning in that using uninitialized values is undefined behavior,
it rather documents the programmer's intent.


