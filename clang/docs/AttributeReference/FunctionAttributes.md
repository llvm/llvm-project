## Function Attributes



### #pragma omp declare simd

{clang-attr-syntaxes}`OMPDeclareSimdDocs`

The `declare simd` construct can be applied to a function to enable the creation
of one or more versions that can process multiple arguments using SIMD
instructions from a single invocation in a SIMD loop. The `declare simd`
directive is a declarative directive. There may be multiple `declare simd`
directives for a function. The use of a `declare simd` construct on a function
enables the creation of SIMD versions of the associated function that can be
used to process multiple arguments from a single invocation from a SIMD loop
concurrently.
The syntax of the `declare simd` construct is as follows:

```none
#pragma omp declare simd [clause[[,] clause] ...] new-line
[#pragma omp declare simd [clause[[,] clause] ...] new-line]
[...]
function definition or declaration
```

where clause is one of the following:

```none
simdlen(length)
linear(argument-list[:constant-linear-step])
aligned(argument-list[:alignment])
uniform(argument-list)
inbranch
notinbranch
```


### #pragma omp declare target

{clang-attr-syntaxes}`OMPDeclareTargetDocs`

The `declare target` directive specifies that variables and functions are mapped
to a device for OpenMP offload mechanism.

The syntax of the declare target directive is as follows:

```c
#pragma omp declare target new-line
declarations-definition-seq
#pragma omp end declare target new-line
```

or

```c
#pragma omp declare target (extended-list) new-line
```

or

```c
#pragma omp declare target clause[ [,] clause ... ] new-line
```

where clause is one of the following:

```c
to(extended-list)
link(list)
device_type(host | nohost | any)
```


### #pragma omp declare variant

{clang-attr-syntaxes}`OMPDeclareVariantDocs`

The `declare variant` directive declares a specialized variant of a base
function and specifies the context in which that specialized variant is used.
The declare variant directive is a declarative directive.
The syntax of the `declare variant` construct is as follows:

```none
#pragma omp declare variant(variant-func-id) clause new-line
[#pragma omp declare variant(variant-func-id) clause new-line]
[...]
function definition or declaration
```

where clause is one of the following:

```none
match(context-selector-specification)
```

and where `variant-func-id` is the name of a function variant that is either a
base language identifier or, for C++, a template-id.

Clang provides the following context selector extensions, used via
`implementation={extension(EXTENSION)}`:

```none
match_all
match_any
match_none
disable_implicit_base
allow_templates
bind_to_declaration
```

The match extensions change when the *entire* context selector is considered a
match for an OpenMP context. The default is `all`, with `none` no trait in the
selector is allowed to be in the OpenMP context, with `any` a single trait in
both the selector and OpenMP context is sufficient. Only a single match
extension trait is allowed per context selector.
The disable extensions remove default effects of the `begin declare variant`
applied to a definition. If `disable_implicit_base` is given, we will not
introduce an implicit base function for a variant if no base function was
found. The variant is still generated but will never be called, due to the
absence of a base function and consequently calls to a base function.
The allow extensions change when the `begin declare variant` effect is
applied to a definition. If `allow_templates` is given, template function
definitions are considered as specializations of existing or assumed template
declarations with the same name. The template parameters for the base functions
are used to instantiate the specialization. If `bind_to_declaration` is given,
apply the same variant rules to function declarations. This allows the user to
override declarations with only a function declaration.


### RootSignature

{clang-attr-syntaxes}`RootSignatureDocs`

The `RootSignature` attribute applies to HLSL entry functions to define what
types of resources are bound to the graphics pipeline.

For details about the use and specification of Root Signatures please see here:
<https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signatures>


### WaveSize

{clang-attr-syntaxes}`WaveSizeDocs`

The `WaveSize` attribute specifies a wave size on a shader entry point in order
to indicate either that a shader depends on or strongly prefers a specific wave
size.
There're 2 versions of the attribute: `WaveSize` and `RangedWaveSize`.
The syntax for `WaveSize` is:

```text
[WaveSize(<numLanes>)]
```

The allowed wave sizes that an HLSL shader may specify are the powers of 2
between 4 and 128, inclusive.
In other words, the set: [4, 8, 16, 32, 64, 128].

The syntax for `RangedWaveSize` is:

```text
[WaveSize(<minWaveSize>, <maxWaveSize>, [prefWaveSize])]
```

Where minWaveSize is the minimum wave size supported by the shader representing
the beginning of the allowed range, maxWaveSize is the maximum wave size
supported by the shader representing the end of the allowed range, and
prefWaveSize is the optional preferred wave size representing the size expected
to be the most optimal for this shader.

`WaveSize` is available for HLSL shader model 6.6 and later.
`RangedWaveSize` available for HLSL shader model 6.8 and later.

The full documentation is available here: <https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html>
and <https://microsoft.github.io/hlsl-specs/proposals/0013-wave-size-range.html>


### _Noreturn

{clang-attr-syntaxes}`C11NoReturnDocs`

A function declared as `_Noreturn` shall not return to its caller. The
compiler will generate a diagnostic for a function declared as `_Noreturn`
that appears to be capable of returning to its caller. Despite being a type
specifier, the `_Noreturn` attribute cannot be specified on a function
pointer type.


### abi_tag

{clang-attr-syntaxes}`AbiTagsDocs`

The `abi_tag` attribute can be applied to a function, variable, class or
inline namespace declaration to modify the mangled name of the entity. It gives
the ability to distinguish between different versions of the same entity but
with different ABI versions supported. For example, a newer version of a class
could have a different set of data members and thus have a different size. Using
the `abi_tag` attribute, it is possible to have different mangled names for
a global variable of the class type. Therefore, the old code could keep using
the old mangled name and the new code will use the new mangled name with tags.


### acquire_capability, acquire_shared_capability

{clang-attr-syntaxes}`AcquireCapabilityDocs`

Marks a function as acquiring a capability.


### alloc_align

{clang-attr-syntaxes}`AllocAlignDocs`

Use `__attribute__((alloc_align(<parameter-index>)))` on a declaration with a
function prototype to specify that the prototype's return value (which must be a
pointer type) is at least as aligned as the value of the indicated parameter.
This includes functions, Objective-C methods, blocks, and declarations of
function pointer, member function pointer, function reference, and block pointer
types. The attribute can also be applied to typedef or type alias declarations
whose underlying type has a function prototype.

The parameter is given by its index in the list of formal parameters; the first
parameter has index 1 unless the function is a C++ non-static member function,
in which case the first parameter has index 2 to account for the implicit `this`
parameter.

```c++
// The returned pointer has the alignment specified by the first parameter.
void *a(size_t align) __attribute__((alloc_align(1)));

// The function pointer's returned pointer has the alignment specified by
// the first parameter of the pointed-to function.
void *(*allocator)(size_t align) __attribute__((alloc_align(1)));

// The returned pointer has the alignment specified by the second parameter.
void *b(void *v, size_t align) __attribute__((alloc_align(2)));

// The returned pointer has the alignment specified by the second visible
// parameter, however it must be adjusted for the implicit 'this' parameter.
void *Foo::b(void *v, size_t align) __attribute__((alloc_align(3)));
```

Note that this attribute merely informs the compiler that a function always
returns a sufficiently aligned pointer. It does not cause the compiler to
emit code to enforce that alignment. The behavior is undefined if the returned
pointer is not sufficiently aligned.


### alloc_size

{clang-attr-syntaxes}`AllocSizeDocs`

The `alloc_size` attribute can be placed on functions that return pointers in
order to hint to the compiler how many bytes of memory will be available at the
returned pointer. `alloc_size` takes one or two arguments.

- `alloc_size(N)` implies that argument number N equals the number of
  available bytes at the returned pointer.
- `alloc_size(N, M)` implies that the product of argument number N and
  argument number M equals the number of available bytes at the returned
  pointer.

Argument numbers are 1-based.

An example of how to use `alloc_size`

```c
void *my_malloc(int a) __attribute__((alloc_size(1)));
void *my_calloc(int a, int b) __attribute__((alloc_size(1, 2)));

int main() {
  void *const p = my_malloc(100);
  assert(__builtin_object_size(p, 0) == 100);
  void *const a = my_calloc(20, 5);
  assert(__builtin_object_size(a, 0) == 100);
}
```

When `-Walloc-size` is enabled, this attribute allows the compiler to
diagnose cases when the allocated memory is insufficient for the size of the
type the returned pointer is cast to.

```c
void *my_malloc(int a) __attribute__((alloc_size(1)));
void consumer_func(int *);

int main() {
  int *ptr = my_malloc(sizeof(int)); // no warning
  int *w = my_malloc(1); // warning: allocation of insufficient size '1' for type 'int' with size '4'
  consumer_func(my_malloc(1)); // warning: allocation of insufficient size '1' for type 'int' with size '4'
}
```

:::{Note}
This attribute works differently in clang than it does in GCC.
Specifically, clang will only trace `const` pointers (as above); we give up
on pointers that are not marked as `const`. In the vast majority of cases,
this is unimportant, because LLVM has support for the `alloc_size`
attribute. However, this may cause mildly unintuitive behavior when used with
other attributes, such as `enable_if`.
:::


### allocator

{clang-attr-syntaxes}`MSAllocatorDocs`

The `__declspec(allocator)` attribute is applied to functions that allocate
memory, such as operator new in C++. When CodeView debug information is emitted
(enabled by `clang -gcodeview` or `clang-cl /Z7`), Clang will attempt to
record the code offset of heap allocation call sites in the debug info. It will
also record the type being allocated using some local heuristics. The Visual
Studio debugger uses this information to [profile memory usage][profile memory usage].

This attribute does not affect optimizations in any way, unlike GCC's
`__attribute__((malloc))`.

[profile memory usage]: https://docs.microsoft.com/en-us/visualstudio/profiling/memory-usage


### always_inline, __force_inline

{clang-attr-syntaxes}`AlwaysInlineDocs`

Inlining heuristics are disabled and inlining is always attempted regardless of
optimization level.

`[[clang::always_inline]]` spelling can be used as a statement attribute; other
spellings of the attribute are not supported on statements. If a statement is
marked `[[clang::always_inline]]` and contains calls, the compiler attempts
to inline those calls.

```c
int example(void) {
  int i;
  [[clang::always_inline]] foo(); // attempts to inline foo
  [[clang::always_inline]] i = bar(); // attempts to inline bar
  [[clang::always_inline]] return f(42, baz(bar())); // attempts to inline everything
}
```

A declaration statement, which is a statement, is not a statement that can have an
attribute associated with it (the attribute applies to the declaration, not the
statement in that case). So this use case will not work:

```c
int example(void) {
  [[clang::always_inline]] int i = bar();
  return i;
}
```

This attribute does not guarantee that inline substitution actually occurs.

:::{Note}
Note: applying this attribute to a coroutine at the `-O0` optimization level
has no effect; other optimization levels may only partially inline and result in a
diagnostic.
:::

See also [the Microsoft Docs on Inline Functions][the microsoft docs on inline functions], [the GCC Common Function
Attribute docs][the gcc common function attribute docs], and [the GCC Inline docs][the gcc inline docs].

[the gcc common function attribute docs]: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html
[the gcc inline docs]: https://gcc.gnu.org/onlinedocs/gcc/Inline.html
[the microsoft docs on inline functions]: https://docs.microsoft.com/en-us/cpp/cpp/inline-functions-cpp


### artificial

{clang-attr-syntaxes}`ArtificialDocs`

The `artificial` attribute can be applied to an inline function. If such a
function is inlined, the attribute indicates that debuggers should associate
the resulting instructions with the call site, rather than with the
corresponding line within the inlined callee.


### assert_capability, assert_shared_capability

{clang-attr-syntaxes}`AssertCapabilityDocs`

Marks a function that dynamically tests whether a capability is held, and halts
the program if it is not held.


### assume

{clang-attr-syntaxes}`OMPAssumeDocs`

Clang supports the `[[omp::assume("assumption")]]` attribute to
provide additional information to the optimizer. The string-literal, here
"assumption", will be attached to the function declaration such that later
analysis and optimization passes can assume the "assumption" to hold.
This is similar to {ref}`__builtin_assume <langext-__builtin_assume>` but
instead of an expression that can be assumed to be non-zero, the assumption is
expressed as a string and it holds for the entire function.

A function can have multiple assume attributes and they propagate from prior
declarations to later definitions. Multiple assumptions are aggregated into a
single comma separated string. Thus, one can provide multiple assumptions via
a comma separated string, i.a.,
`[[omp::assume("assumption1,assumption2")]]`.

While LLVM plugins might provide more assumption strings, the default LLVM
optimization passes are aware of the following assumptions:

```none
"omp_no_openmp"
"omp_no_openmp_routines"
"omp_no_parallelism"
"omp_no_openmp_constructs"
```

The OpenMP standard defines the meaning of OpenMP assumptions ("omp_XYZ" is
spelled "XYZ" in the [OpenMP 5.1 Standard][openmp 5.1 standard]).

[openmp 5.1 standard]: https://www.openmp.org/spec-html/5.1/openmpsu37.html#x56-560002.5.2


### assume_aligned

{clang-attr-syntaxes}`AssumeAlignedDocs`

Use `__attribute__((assume_aligned(<alignment>[,<offset>]))` on a function
declaration to specify that the return value of the function (which must be a
pointer type) has the specified offset, in bytes, from an address with the
specified alignment. The offset is taken to be zero if omitted.

```c++
// The returned pointer value has 32-byte alignment.
void *a() __attribute__((assume_aligned (32)));

// The returned pointer value is 4 bytes greater than an address having
// 32-byte alignment.
void *b() __attribute__((assume_aligned (32, 4)));
```

Note that this attribute provides information to the compiler regarding a
condition that the code already ensures is true. It does not cause the compiler
to enforce the provided alignment assumption.


### availability

{clang-attr-syntaxes}`AvailabilityDocs`

The `availability` attribute can be placed on declarations to describe the
lifecycle of that declaration relative to operating system versions. Consider
the function declaration for a hypothetical function `f`:

```c++
void f(void) __attribute__((availability(macos,introduced=10.4,deprecated=10.6,obsoleted=10.7)));
```

The availability attribute states that `f` was introduced in macOS 10.4,
deprecated in macOS 10.6, and obsoleted in macOS 10.7. This information
is used by Clang to determine when it is safe to use `f`: for example, if
Clang is instructed to compile code for macOS 10.5, a call to `f()`
succeeds. If Clang is instructed to compile code for macOS 10.6, the call
succeeds but Clang emits a warning specifying that the function is deprecated.
Finally, if Clang is instructed to compile code for macOS 10.7, the call
fails because `f()` is no longer available.

Clang is instructed to compile code for a minimum deployment version using
the `-target` or `-mtargetos` command line arguments. For example,
macOS 10.7 would be specified as `-target x86_64-apple-macos10.7` or
`-mtargetos=macos10.7`. Variants like Mac Catalyst are specified as
`-target arm64-apple-ios15.0-macabi` or `-mtargetos=ios15.0-macabi`

The availability attribute is a comma-separated list starting with the
platform name and then including clauses specifying important milestones in the
declaration's lifetime (in any order) along with additional information. Those
clauses can be:

introduced=*version*

: The first version in which this declaration was introduced.

deprecated=*version*

: The first version in which this declaration was deprecated, meaning that
  users should migrate away from this API.

obsoleted=*version*

: The first version in which this declaration was obsoleted, meaning that it
  was removed completely and can no longer be used.

unavailable

: This declaration is never available on this platform.

message=*string-literal*

: Additional message text that Clang will provide when emitting a warning or
  error about use of a deprecated or obsoleted declaration. Useful to direct
  users to replacement APIs.

replacement=*string-literal*

: Additional message text that Clang will use to provide Fix-It when emitting
  a warning about use of a deprecated declaration. The Fix-It will replace
  the deprecated declaration with the new declaration specified.

environment=*identifier*

: Target environment in which this declaration is available. If present,
  the availability attribute applies only to targets with the same platform
  and environment. The parameter is currently supported only in HLSL.

Multiple availability attributes can be placed on a declaration, which may
correspond to different platforms. For most platforms, the availability
attribute with the platform corresponding to the target platform will be used;
any others will be ignored. However, the availability for `watchOS` and
`tvOS` can be implicitly inferred from an `iOS` availability attribute.
Any explicit availability attributes for those platforms are still preferred over
the implicitly inferred availability attributes. If no availability attribute
specifies availability for the current target platform, the availability
attributes are ignored. Supported platforms are:

`iOS`
`macOS`
`tvOS`
`watchOS`
`iOSApplicationExtension`
`macOSApplicationExtension`
`tvOSApplicationExtension`
`watchOSApplicationExtension`
`macCatalyst`
`macCatalystApplicationExtension`
`visionOS`
`visionOSApplicationExtension`
`driverkit`
`anyAppleOS`
`swift`
`android`
`fuchsia`
`ohos`
`zos`
`ShaderModel`

Some platforms have alias names:

`ios`
`macos`
`macosx (deprecated)`
`tvos`
`watchos`
`ios_app_extension`
`macos_app_extension`
`macosx_app_extension (deprecated)`
`tvos_app_extension`
`watchos_app_extension`
`maccatalyst`
`maccatalyst_app_extension`
`visionos`
`visionos_app_extension`
`anyappleos`
`shadermodel`

Supported environment names for the ShaderModel platform:

`pixel`
`vertex`
`geometry`
`hull`
`domain`
`compute`
`raygeneration`
`intersection`
`anyhit`
`closesthit`
`miss`
`callable`
`mesh`
`amplification`
`library`

The special platform `anyAppleOS` (alias: `anyappleos`) is a shorthand that
applies the availability attribute to all Apple Darwin platforms. An explicit
platform-specific availability attribute takes precedence over an `anyAppleOS`
attribute for that platform. Versions specified with `anyAppleOS` must be at
least 26.0, which is the first OS release where all supported Apple platforms
share a unified version number.

A declaration can typically be used even when deploying back to a platform
version prior to when the declaration was introduced. When this happens, the
declaration is [weakly linked](https://developer.apple.com/library/mac/#documentation/MacOSX/Conceptual/BPFrameworks/Concepts/WeakLinking.html),
as if the `weak_import` attribute were added to the declaration. A
weakly-linked declaration may or may not be present a run-time, and a program
can determine whether the declaration is present by checking whether the
address of that declaration is non-NULL.

The flag `strict` disallows using API when deploying back to a
platform version prior to when the declaration was introduced. An
attempt to use such API before its introduction causes a hard error.
Weakly-linking is almost always a better API choice, since it allows
users to query availability at runtime.

If there are multiple declarations of the same entity, the availability
attributes must either match on a per-platform basis or later
declarations must not have availability attributes for that
platform. For example:

```c
void g(void) __attribute__((availability(macos,introduced=10.4)));
void g(void) __attribute__((availability(macos,introduced=10.4))); // okay, matches
void g(void) __attribute__((availability(ios,introduced=4.0))); // okay, adds a new platform
void g(void); // okay, inherits both macos and ios availability from above.
void g(void) __attribute__((availability(macos,introduced=10.5))); // error: mismatch
```

When one method overrides another, the overriding method can be more widely available than the overridden method, e.g.,:

```objc
@interface A
- (id)method __attribute__((availability(macos,introduced=10.4)));
- (id)method2 __attribute__((availability(macos,introduced=10.4)));
@end

@interface B : A
- (id)method __attribute__((availability(macos,introduced=10.3))); // okay: method moved into base class later
- (id)method __attribute__((availability(macos,introduced=10.5))); // error: this method was available via the base class in 10.4
@end
```

Starting with the macOS 10.12 SDK, the `API_AVAILABLE` macro from
`<os/availability.h>` can simplify the spelling:

```objc
@interface A
- (id)method API_AVAILABLE(macos(10.11)));
- (id)otherMethod API_AVAILABLE(macos(10.11), ios(11.0));
@end
```

Availability attributes can also be applied using a `#pragma clang attribute`.
Any explicit availability attribute whose platform corresponds to the target
platform is applied to a declaration regardless of the availability attributes
specified in the pragma. For example, in the code below,
`hasExplicitAvailabilityAttribute` will use the `macOS` availability
attribute that is specified with the declaration, whereas
`getsThePragmaAvailabilityAttribute` will use the `macOS` availability
attribute that is applied by the pragma.

```c
#pragma clang attribute push (__attribute__((availability(macOS, introduced=10.12))), apply_to=function)
void getsThePragmaAvailabilityAttribute(void);
void hasExplicitAvailabilityAttribute(void) __attribute__((availability(macos,introduced=10.4)));
#pragma clang attribute pop
```

For platforms like `watchOS` and `tvOS`, whose availability attributes can
be implicitly inferred from an `iOS` availability attribute, the logic is
slightly more complex. The explicit and the pragma-applied availability
attributes whose platform corresponds to the target platform are applied as
described in the previous paragraph. However, the implicitly inferred attributes
are applied to a declaration only when there is no explicit or pragma-applied
availability attribute whose platform corresponds to the target platform. For
example, the function below will receive the `tvOS` availability from the
pragma rather than using the inferred `iOS` availability from the declaration:

```c
#pragma clang attribute push (__attribute__((availability(tvOS, introduced=12.0))), apply_to=function)
void getsThePragmaTVOSAvailabilityAttribute(void) __attribute__((availability(iOS,introduced=11.0)));
#pragma clang attribute pop
```

The compiler is also able to apply implicitly inferred attributes from a pragma
as well. For example, when targeting `tvOS`, the function below will receive
a `tvOS` availability attribute that is implicitly inferred from the `iOS`
availability attribute applied by the pragma:

```c
#pragma clang attribute push (__attribute__((availability(iOS, introduced=12.0))), apply_to=function)
void infersTVOSAvailabilityFromPragma(void);
#pragma clang attribute pop
```

The implicit attributes that are inferred from explicitly specified attributes
whose platform corresponds to the target platform are applied to the declaration
even if there is an availability attribute that can be inferred from a pragma.
For example, the function below will receive the `tvOS, introduced=11.0`
availability that is inferred from the attribute on the declaration rather than
inferring availability from the pragma:

```c
#pragma clang attribute push (__attribute__((availability(iOS, unavailable))), apply_to=function)
void infersTVOSAvailabilityFromAttributeNextToDeclaration(void)
  __attribute__((availability(iOS,introduced=11.0)));
#pragma clang attribute pop
```

Also see the documentation for
{ref}`@available <langext-objective-c-available>`


### btf_decl_tag

{clang-attr-syntaxes}`BTFDeclTagDocs`

Clang supports the `__attribute__((btf_decl_tag("ARGUMENT")))` attribute for
all targets. This attribute may be attached to a struct/union, struct/union
field, function, function parameter, variable or typedef declaration. If -g is
specified, the `ARGUMENT` info will be preserved in IR and be emitted to
dwarf. For BPF targets, the `ARGUMENT` info will be emitted to .BTF ELF
section too.


### callback

{clang-attr-syntaxes}`CallbackDocs`

The `callback` attribute specifies that the annotated function may invoke the
specified callback zero or more times. The callback, as well as the passed
arguments, are identified by their parameter name or position (starting with
1!) in the annotated function. The first position in the attribute identifies
the callback callee, the following positions declare describe its arguments.
The callback callee is required to be callable with the number, and order, of
the specified arguments. The index `0`, or the identifier `this`, is used to
represent an implicit "this" pointer in class methods. If there is no implicit
"this" pointer it shall not be referenced. The index `-1`, or the name `__`,
represents an unknown callback callee argument. This can be a value which is
not present in the declared parameter list, or one that is, but is potentially
inspected, captured, or modified. Parameter names and indices can be mixed in
the callback attribute.

The `callback` attribute, which is directly translated to `callback`
metadata (<http://llvm.org/docs/LangRef.html#callback-metadata>), make the
connection between the call to the annotated function and the callback callee.
This can enable interprocedural optimizations which were otherwise impossible.
If a function parameter is mentioned in the `callback` attribute, through its
position, it is undefined if that parameter is used for anything other than the
actual callback. Inspected, captured, or modified parameters shall not be
listed in the `callback` metadata.

Example encodings for the callback performed by `pthread_create` are shown
below. The explicit attribute annotation indicates that the third parameter
(`start_routine`) is called zero or more times by the `pthread_create` function,
and that the fourth parameter (`arg`) is passed along. Note that the callback
behavior of `pthread_create` is automatically recognized by Clang. In addition,
the declarations of `__kmpc_fork_teams` and `__kmpc_fork_call`, generated for
`#pragma omp target teams` and `#pragma omp parallel`, respectively, are also
automatically recognized as broker functions. Further functions might be added
in the future.

```c
__attribute__((callback (start_routine, arg)))
int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine) (void *), void *arg);

__attribute__((callback (3, 4)))
int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine) (void *), void *arg);
```


### cf_consumed, cf_returns_not_retained, cf_returns_retained, ns_consumed, ns_consumes_self, ns_returns_autoreleased, ns_returns_not_retained, ns_returns_retained, os_consumed, os_consumes_this, os_returns_not_retained, os_returns_retained, os_returns_retained_on_non_zero, os_returns_retained_on_zero

{clang-attr-syntaxes}`RetainBehaviorDocs`

The behavior of a function with respect to reference counting for Foundation
(Objective-C), CoreFoundation (C) and OSObject (C++) is determined by a naming
convention (e.g. functions starting with "get" are assumed to return at
`+0`).

It can be overridden using a family of the following attributes. In
Objective-C, the annotation `__attribute__((ns_returns_retained))` applied to
a function communicates that the object is returned at `+1`, and the caller
is responsible for freeing it.
Similarly, the annotation `__attribute__((ns_returns_not_retained))`
specifies that the object is returned at `+0` and the ownership remains with
the callee.
The annotation `__attribute__((ns_consumes_self))` specifies that
the Objective-C method call consumes the reference to `self`, e.g. by
attaching it to a supplied parameter.
Additionally, parameters can have an annotation
`__attribute__((ns_consumed))`, which specifies that passing an owned object
as that parameter effectively transfers the ownership, and the caller is no
longer responsible for it.
These attributes affect code generation when interacting with ARC code, and
they are used by the Clang Static Analyzer.

In C programs using CoreFoundation, a similar set of attributes:
`__attribute__((cf_returns_not_retained))`,
`__attribute__((cf_returns_retained))` and `__attribute__((cf_consumed))`
have the same respective semantics when applied to CoreFoundation objects.
These attributes affect code generation when interacting with ARC code, and
they are used by the Clang Static Analyzer.

(os-retained-attr-family)=

Finally, in C++ interacting with XNU kernel (objects inheriting from OSObject),
the same attribute family is present:
`__attribute__((os_returns_not_retained))`,
`__attribute__((os_returns_retained))` and `__attribute__((os_consumed))`,
with the same respective semantics.
Similar to `__attribute__((ns_consumes_self))`,
`__attribute__((os_consumes_this))` specifies that the method call consumes
the reference to "this" (e.g., when attaching it to a different object supplied
as a parameter).
Out parameters (parameters the function is meant to write into,
either via pointers-to-pointers or references-to-pointers)
may be annotated with `__attribute__((os_returns_retained))`
or `__attribute__((os_returns_not_retained))` which specifies that the object
written into the out parameter should (or respectively should not) be released
after use.
Since often out parameters may or may not be written depending on the exit
code of the function,
annotations `__attribute__((os_returns_retained_on_zero))`
and `__attribute__((os_returns_retained_on_non_zero))` specify that
an out parameter at `+1` is written if and only if the function returns a zero
(respectively non-zero) error code.
Observe that return-code-dependent out parameter annotations are only
available for retained out parameters, as non-retained object do not have to be
released by the callee.
These attributes are only used by the Clang Static Analyzer.

The family of attributes `X_returns_X_retained` can be added to functions,
C++ methods, and Objective-C methods and properties.
Attributes `X_consumed` can be added to parameters of methods, functions,
and Objective-C methods.


(langext-cfi_canonical_jump_table)=

### cfi_canonical_jump_table

{clang-attr-syntaxes}`CFICanonicalJumpTableDocs`

Use `__attribute__((cfi_canonical_jump_table))` on a function declaration to
make the function's CFI jump table canonical. See {ref}`the CFI documentation
<cfi-canonical-jump-tables>` for more details.


(langext-cfi_salt)=

### cfi_salt

{clang-attr-syntaxes}`CFISaltDocs`

The `cfi_salt` attribute specifies a string literal that is used as a salt
for Control-Flow Integrity (CFI) checks to distinguish between functions with
the same type signature. This attribute can be applied to function declarations,
function definitions, and function pointer typedefs.

The attribute prevents function pointers from being replaced with pointers to
functions that have a compatible type, which can be a CFI bypass vector.

**Syntax:**

- GNU-style: `__attribute__((cfi_salt("<salt_string>")))`
- C++11-style: `[[clang::cfi_salt("<salt_string>")]]`

**Usage:**

The attribute takes a single string literal argument that serves as the salt.
Functions or function types with different salt values will have different CFI
hashes, even if they have identical type signatures.

**Motivation:**

In large codebases like the Linux kernel, there are often hundreds of functions
with identical type signatures that are called indirectly:

```
1662 functions with void (*)(void)
1179 functions with int (*)(void)
 ...
```

By salting the CFI hashes, you can make CFI more robust by ensuring that
functions intended for different purposes have distinct CFI identities.

**Type Compatibility:**

- Functions with different salt values are considered to have incompatible types
- Function pointers with different salt values cannot be assigned to each other
- All declarations of the same function must use the same salt value

**Example:**

```c
// Header file - define convenience macros
#define __cfi_salt(s) __attribute__((cfi_salt(s)))

// Typedef for regular function pointers
typedef int (*fptr_t)(void);

// Typedef for salted function pointers
typedef int (*fptr_salted_t)(void) __cfi_salt("pepper");

struct widget_ops {
  fptr_t init;          // Regular CFI
  fptr_salted_t exec;   // Salted CFI
  fptr_t cleanup;       // Regular CFI
};

// Function implementations
static int widget_init(void) { return 0; }
static int widget_exec(void) __cfi_salt("pepper") { return 1; }
static int widget_cleanup(void) { return 0; }

static struct widget_ops ops = {
  .init = widget_init,      // OK - compatible types
  .exec = widget_exec,      // OK - both use "pepper" salt
  .cleanup = widget_cleanup // OK - compatible types
};

// Using C++11 attribute syntax
void secure_callback(void) [[clang::cfi_salt("secure")]];

// This would cause a compilation error:
// fptr_t bad_ptr = widget_exec;  // Error: incompatible types
```

**Notes:**

- The salt string can contain non-NULL ASCII characters, including spaces and
  quotes
- This attribute only applies to function types; using it on non-function
  types will generate a warning
- All declarations and definitions of the same function must use identical
  salt values
- The attribute affects type compatibility during compilation and CFI hash
  generation during code generation


### clang::builtin_alias, clang_builtin_alias

{clang-attr-syntaxes}`BuiltinAliasDocs`

This attribute is used in the implementation of the C intrinsics.
It allows the C intrinsic functions to be declared using the names defined
in target builtins, and still be recognized as clang builtins equivalent to the
underlying name. For example, `riscv_vector.h` declares the function `vadd`
with `__attribute__((clang_builtin_alias(__builtin_rvv_vadd_vv_i8m1)))`.
This ensures that both functions are recognized as that clang builtin,
and in the latter case, the choice of which builtin to identify the
function as can be deferred until after overload resolution.

This attribute can only be used to set up the aliases for certain ARM/RISC-V
C intrinsic functions; it is intended for use only inside `arm_*.h` and
`riscv_*.h` and is not a general mechanism for declaring arbitrary aliases
for clang builtin functions.


### clang_arm_builtin_alias

{clang-attr-syntaxes}`ArmBuiltinAliasDocs`

This attribute is used in the implementation of the ACLE intrinsics.
It allows the intrinsic functions to
be declared using the names defined in ACLE, and still be recognized
as clang builtins equivalent to the underlying name. For example,
`arm_mve.h` declares the function `vaddq_u32` with
`__attribute__((__clang_arm_mve_alias(__builtin_arm_mve_vaddq_u32)))`,
and similarly, one of the type-overloaded declarations of `vaddq`
will have the same attribute. This ensures that both functions are
recognized as that clang builtin, and in the latter case, the choice
of which builtin to identify the function as can be deferred until
after overload resolution.

This attribute can only be used to set up the aliases for certain Arm
intrinsic functions; it is intended for use only inside `arm_*.h`
and is not a general mechanism for declaring arbitrary aliases for
clang builtin functions.

In order to avoid duplicating the attribute definitions for similar
purpose for other architecture, there is a general form for the
attribute `clang_builtin_alias`.


### clspv_libclc_builtin

{clang-attr-syntaxes}`ClspvLibclcBuiltinDoc`

Attribute used by [clspv][clspv] (OpenCL-C to Vulkan SPIR-V compiler) to identify functions coming from [libclc][libclc] (OpenCL-C builtin library).

```c
void __attribute__((clspv_libclc_builtin)) libclc_builtin() {}
```

[clspv]: https://github.com/google/clspv
[libclc]: https://libclc.llvm.org


### cmse_nonsecure_entry

{clang-attr-syntaxes}`ArmCmseNSEntryDocs`

This attribute declares a function that can be called from non-secure state, or
from secure state. Entering from and returning to non-secure state would switch
to and from secure state, respectively, and prevent flow of information
to non-secure state, except via return values. See [ARMv8-M Security Extensions:
Requirements on Development Tools - Engineering Specification Documentation](https://developer.arm.com/docs/ecm0359818/latest/) for more information.


### code_seg

{clang-attr-syntaxes}`CodeSegDocs`

The `__declspec(code_seg)` attribute enables the placement of code into separate
named segments that can be paged or locked in memory individually. This attribute
is used to control the placement of instantiated templates and compiler-generated
code. See the documentation for [`__declspec(code_seg)`][__declspec(code_seg)] on MSDN.

[__declspec(code_seg)]: http://msdn.microsoft.com/en-us/library/dn636922.aspx


### cold

{clang-attr-syntaxes}`ColdFunctionEntryDocs`

`__attribute__((cold))` marks a function as cold, as a manual alternative to PGO hotness data.
If PGO data is available, the profile count based hotness overrides the `__attribute__((cold))` annotation (unlike `__attribute__((hot))`).


### constant_id

{clang-attr-syntaxes}`VkConstantIdDocs`

The `vk::constant_id` attribute specifies the id for a SPIR-V specialization
constant. The attribute applies to const global scalar variables. The variable must be initialized with a C++11 constexpr.
In SPIR-V, the
variable will be replaced with an `OpSpecConstant` with the given id.
The syntax is:

```text
[[vk::constant_id(<Id>)]] const T Name = <Init>
```


### constructor, destructor

{clang-attr-syntaxes}`CtorDtorDocs`

The `constructor` attribute causes the function to be called before entering
`main()`, and the `destructor` attribute causes the function to be called
after returning from `main()` or when the `exit()` function has been
called. Note, `quick_exit()`, `_Exit()`, and `abort()` prevent a function
marked `destructor` from being called.

The constructor or destructor function should not accept any arguments and its
return type should be `void`.

The attributes accept an optional argument used to specify the priority order
in which to execute constructor and destructor functions. The priority is
given as an integer constant expression between 101 and 65535 (inclusive).
Priorities outside of that range are reserved for use by the implementation. A
lower value indicates a higher priority of initialization. Note that only the
relative ordering of values is important. For example:

```c++
__attribute__((constructor(200))) void foo(void);
__attribute__((constructor(101))) void bar(void);
```

`bar()` will be called before `foo()`, and both will be called before
`main()`. If no argument is given to the `constructor` or `destructor`
attribute, they default to the value `65535`.


### convergent

{clang-attr-syntaxes}`ConvergentDocs`

The `convergent` attribute can be placed on a function declaration. It is
translated into the LLVM `convergent` attribute, which indicates that the call
instructions of a function with this attribute cannot be made control-dependent
on any additional values.

This attribute is different from `noduplicate` because it allows duplicating
function calls if it can be proved that the duplicated function calls are
not made control-dependent on any additional values, e.g., unrolling a loop
executed by all work items.

Sample usage:

```c
void convfunc(void) __attribute__((convergent));
// Setting it as a C++11 attribute is also valid in a C++ program.
// void convfunc(void) [[clang::convergent]];
```


### cpu_dispatch, cpu_specific

{clang-attr-syntaxes}`CPUSpecificCPUDispatchDocs`

The `cpu_specific` and `cpu_dispatch` attributes are used to define and
resolve multiversioned functions. This form of multiversioning provides a
mechanism for declaring versions across translation units and manually
specifying the resolved function list. A specified CPU defines a set of minimum
features that are required for the function to be called. The result of this is
that future processors execute the most restrictive version of the function the
new processor can execute.

In addition, unlike the ICC implementation of this feature, the selection of the
version does not consider the manufacturer or microarchitecture of the processor.
It tests solely the list of features that are both supported by the specified
processor and present in the compiler-rt library. This can be surprising at times,
as the runtime processor may be from a completely different manufacturer, as long
as it supports the same feature set.

This can additionally be surprising, as some processors are indistringuishable from
others based on the list of testable features. When this happens, the variant
is selected in an unspecified manner.

Function versions are defined with `cpu_specific`, which takes one or more CPU
names as a parameter. For example:

```c
// Declares and defines the ivybridge version of single_cpu.
__attribute__((cpu_specific(ivybridge)))
void single_cpu(void){}

// Declares and defines the atom version of single_cpu.
__attribute__((cpu_specific(atom)))
void single_cpu(void){}

// Declares and defines both the ivybridge and atom version of multi_cpu.
__attribute__((cpu_specific(ivybridge, atom)))
void multi_cpu(void){}
```

A dispatching (or resolving) function can be declared anywhere in a project's
source code with `cpu_dispatch`. This attribute takes one or more CPU names
as a parameter (like `cpu_specific`). Functions marked with `cpu_dispatch`
are not expected to be defined, only declared. If such a marked function has a
definition, any side effects of the function are ignored; trivial function
bodies are permissible for ICC compatibility.

```c
// Creates a resolver for single_cpu above.
__attribute__((cpu_dispatch(ivybridge, atom)))
void single_cpu(void){}

// Creates a resolver for multi_cpu, but adds a 3rd version defined in another
// translation unit.
__attribute__((cpu_dispatch(ivybridge, atom, sandybridge)))
void multi_cpu(void){}
```

Note that it is possible to have a resolving function that dispatches based on
more or fewer options than are present in the program. Specifying fewer will
result in the omitted options not being considered during resolution. Specifying
a version for resolution that isn't defined in the program will result in a
linking failure.

It is also possible to specify a CPU name of `generic` which will be resolved
if the executing processor doesn't satisfy the features required in the CPU
name. The behavior of a program executing on a processor that doesn't satisfy
any option of a multiversioned function is undefined.


### device_kernel, nvptx_kernel, amdgpu_kernel, kernel, __kernel

{clang-attr-syntaxes}`DeviceKernelDocs`

These attributes specify that the function represents a kernel for device offloading.
The specific semantics depend on the offloading language, target, and attribute spelling.
Here is a code example using the attribute to mark a function as a kernel:

```c++
[[clang::device_kernel]] int foo(int x) { return ++x; }
```


### diagnose_as_builtin

{clang-attr-syntaxes}`DiagnoseAsBuiltinDocs`

The `diagnose_as_builtin` attribute indicates that Fortify diagnostics are to
be applied to the declared function as if it were the function specified by the
attribute. The builtin function whose diagnostics are to be mimicked should be
given. In addition, the order in which arguments should be applied must also
be given.

For example, the attribute can be used as follows.

```c
__attribute__((diagnose_as_builtin(__builtin_memset, 3, 2, 1)))
void *mymemset(int n, int c, void *s) {
  // ...
}
```

This indicates that calls to `mymemset` should be diagnosed as if they were
calls to `__builtin_memset`. The arguments `3, 2, 1` indicate by index the
order in which arguments of `mymemset` should be applied to
`__builtin_memset`. The third argument should be applied first, then the
second, and then the first. Thus (when Fortify warnings are enabled) the call
`mymemset(n, c, s)` will diagnose overflows as if it were the call
`__builtin_memset(s, c, n)`.

For variadic functions, the variadic arguments must come in the same order as
they would to the builtin function, after all normal arguments. For instance,
to diagnose a new function as if it were `sscanf`, we can use the attribute as
follows.

```c
__attribute__((diagnose_as_builtin(sscanf, 1, 2)))
int mysscanf(const char *str, const char *format, ...)  {
  // ...
}
```

Then the call `mysscanf("abc def", "%4s %4s", buf1, buf2)` will be diagnosed as
if it were the call `sscanf("abc def", "%4s %4s", buf1, buf2)`.

This attribute cannot be applied to non-static member functions.


### diagnose_if

{clang-attr-syntaxes}`DiagnoseIfDocs`

The `diagnose_if` attribute can be placed on function declarations to emit
warnings or errors at compile-time if calls to the attributed function meet
certain user-defined criteria. For example:

```c
int abs(int a)
  __attribute__((diagnose_if(a >= 0, "Redundant abs call", "warning")));
int must_abs(int a)
  __attribute__((diagnose_if(a >= 0, "Redundant abs call", "error")));

int val = abs(1); // warning: Redundant abs call
int val2 = must_abs(1); // error: Redundant abs call
int val3 = abs(val);
int val4 = must_abs(val); // Because run-time checks are not emitted for
                          // diagnose_if attributes, this executes without
                          // issue.
```

`diagnose_if` is closely related to `enable_if`, with a few key differences:

- Overload resolution is not aware of `diagnose_if` attributes: they're
  considered only after we select the best candidate from a given candidate set.
- Function declarations that differ only in their `diagnose_if` attributes are
  considered to be redeclarations of the same function (not overloads).
- If the condition provided to `diagnose_if` cannot be evaluated, no
  diagnostic will be emitted.

Otherwise, `diagnose_if` is essentially the logical negation of `enable_if`.

As a result of bullet number two, `diagnose_if` attributes will stack on the
same function. For example:

```c
int foo() __attribute__((diagnose_if(1, "diag1", "warning")));
int foo() __attribute__((diagnose_if(1, "diag2", "warning")));

int bar = foo(); // warning: diag1
                 // warning: diag2
int (*fooptr)(void) = foo; // warning: diag1
                           // warning: diag2

constexpr int supportsAPILevel(int N) { return N < 5; }
int baz(int a)
  __attribute__((diagnose_if(!supportsAPILevel(10),
                             "Upgrade to API level 10 to use baz", "error")));
int baz(int a)
  __attribute__((diagnose_if(!a, "0 is not recommended.", "warning")));

int (*bazptr)(int) = baz; // error: Upgrade to API level 10 to use baz
int v = baz(0); // error: Upgrade to API level 10 to use baz
```

Query for this feature with `__has_attribute(diagnose_if)`.


### disable_sanitizer_instrumentation

{clang-attr-syntaxes}`DisableSanitizerInstrumentationDocs`

Use the `disable_sanitizer_instrumentation` attribute on a function,
Objective-C method, or global variable, to specify that no sanitizer
instrumentation should be applied.

This is not the same as `__attribute__((no_sanitize(...)))`, which depending
on the tool may still insert instrumentation to prevent false positive reports.


### disable_tail_calls

{clang-attr-syntaxes}`DisableTailCallsDocs`

The `disable_tail_calls` attribute instructs the backend to not perform tail
call optimization inside the marked function.

For example:

```c
int callee(int);

int foo(int a) __attribute__((disable_tail_calls)) {
  return callee(a); // This call is not tail-call optimized.
}
```

Marking virtual functions as `disable_tail_calls` is legal.

```c++
int callee(int);

class Base {
public:
  [[clang::disable_tail_calls]] virtual int foo1() {
    return callee(); // This call is not tail-call optimized.
  }
};

class Derived1 : public Base {
public:
  int foo1() override {
    return callee(); // This call is tail-call optimized.
  }
};
```


### enable_if

{clang-attr-syntaxes}`EnableIfDocs`

:::{Note}
Some features of this attribute are experimental. The meaning of
multiple enable_if attributes on a single declaration is subject to change in
a future version of clang. Also, the ABI is not standardized and the name
mangling may change in future versions. To avoid that, use asm labels.
:::

The `enable_if` attribute can be placed on function declarations to control
which overload is selected based on the values of the function's arguments.
When combined with the `overloadable` attribute, this feature is also
available in C.

```c++
int isdigit(int c);
int isdigit(int c)
  __attribute__((enable_if(c <= -1 || c > 255, "chosen when 'c' is out of range")))
  __attribute__((unavailable("'c' must have the value of an unsigned char or EOF")));

void foo(char c) {
  isdigit(c);
  isdigit(10);
  isdigit(-10);  // results in a compile-time error.
}
```

The enable_if attribute takes two arguments, the first is an expression written
in terms of the function parameters, the second is a string explaining why this
overload candidate could not be selected to be displayed in diagnostics. The
expression is part of the function signature for the purposes of determining
whether it is a redeclaration (following the rules used when determining
whether a C++ template specialization is ODR-equivalent), but is not part of
the type.

The enable_if expression is evaluated as if it were the body of a
bool-returning constexpr function declared with the arguments of the function
it is being applied to, then called with the parameters at the call site. If the
result is false or could not be determined through constant expression
evaluation, then this overload will not be chosen and the provided string may
be used in a diagnostic if the compile fails as a result.

Because the enable_if expression is an unevaluated context, there are no global
state changes, nor the ability to pass information from the enable_if
expression to the function body. For example, suppose we want calls to
strnlen(strbuf, maxlen) to resolve to strnlen_chk(strbuf, maxlen, size of
strbuf) only if the size of strbuf can be determined:

```c++
__attribute__((always_inline))
static inline size_t strnlen(const char *s, size_t maxlen)
  __attribute__((overloadable))
  __attribute__((enable_if(__builtin_object_size(s, 0) != -1))),
                           "chosen when the buffer size is known but 'maxlen' is not")))
{
  return strnlen_chk(s, maxlen, __builtin_object_size(s, 0));
}
```

Multiple enable_if attributes may be applied to a single declaration. In this
case, the enable_if expressions are evaluated from left to right in the
following manner. First, the candidates whose enable_if expressions evaluate to
false or cannot be evaluated are discarded. If the remaining candidates do not
share ODR-equivalent enable_if expressions, the overload resolution is
ambiguous. Otherwise, enable_if overload resolution continues with the next
enable_if attribute on the candidates that have not been discarded and have
remaining enable_if attributes. In this way, we pick the most specific
overload out of a number of viable overloads using enable_if.

```c++
void f() __attribute__((enable_if(true, "")));  // #1
void f() __attribute__((enable_if(true, ""))) __attribute__((enable_if(true, "")));  // #2

void g(int i, int j) __attribute__((enable_if(i, "")));  // #1
void g(int i, int j) __attribute__((enable_if(j, ""))) __attribute__((enable_if(true)));  // #2
```

In this example, a call to f() is always resolved to #2, as the first enable_if
expression is ODR-equivalent for both declarations, but #1 does not have another
enable_if expression to continue evaluating, so the next round of evaluation has
only a single candidate. In a call to g(1, 1), the call is ambiguous even though
#2 has more enable_if attributes, because the first enable_if expressions are
not ODR-equivalent.

Query for this feature with `__has_attribute(enable_if)`.

Note that functions with one or more `enable_if` attributes may not have
their address taken, unless all of the conditions specified by said
`enable_if` are constants that evaluate to `true`. For example:

```c
const int TrueConstant = 1;
const int FalseConstant = 0;
int f(int a) __attribute__((enable_if(a > 0, "")));
int g(int a) __attribute__((enable_if(a == 0 || a != 0, "")));
int h(int a) __attribute__((enable_if(1, "")));
int i(int a) __attribute__((enable_if(TrueConstant, "")));
int j(int a) __attribute__((enable_if(FalseConstant, "")));

void fn() {
  int (*ptr)(int);
  ptr = &f; // error: 'a > 0' is not always true
  ptr = &g; // error: 'a == 0 || a != 0' is not a truthy constant
  ptr = &h; // OK: 1 is a truthy constant
  ptr = &i; // OK: 'TrueConstant' is a truthy constant
  ptr = &j; // error: 'FalseConstant' is a constant, but not truthy
}
```

Because `enable_if` evaluation happens during overload resolution,
`enable_if` may give unintuitive results when used with templates, depending
on when overloads are resolved. In the example below, clang will emit a
diagnostic about no viable overloads for `foo` in `bar`, but not in `baz`:

```c++
double foo(int i) __attribute__((enable_if(i > 0, "")));
void *foo(int i) __attribute__((enable_if(i <= 0, "")));
template <int I>
auto bar() { return foo(I); }

template <typename T>
auto baz() { return foo(T::number); }

struct WithNumber { constexpr static int number = 1; };
void callThem() {
  bar<sizeof(WithNumber)>();
  baz<WithNumber>();
}
```

This is because, in `bar`, `foo` is resolved prior to template
instantiation, so the value for `I` isn't known (thus, both `enable_if`
conditions for `foo` fail). However, in `baz`, `foo` is resolved during
template instantiation, so the value for `T::number` is known.


### enforce_tcb

{clang-attr-syntaxes}`EnforceTCBDocs`

The `enforce_tcb` attribute can be placed on functions to enforce that a
trusted compute base (TCB) does not call out of the TCB. This generates a
warning every time a function not marked with an `enforce_tcb` attribute is
called from a function with the `enforce_tcb` attribute. A function may be a
part of multiple TCBs. Invocations through function pointers are currently
not checked. Builtins are considered to a part of every TCB.

- `enforce_tcb(Name)` indicates that this function is a part of the TCB named `Name`


### enforce_tcb_leaf

{clang-attr-syntaxes}`EnforceTCBLeafDocs`

The `enforce_tcb_leaf` attribute satisfies the requirement enforced by
`enforce_tcb` for the marked function to be in the named TCB but does not
continue to check the functions called from within the leaf function.

- `enforce_tcb_leaf(Name)` indicates that this function is a part of the TCB named `Name`


### error, warning

{clang-attr-syntaxes}`ErrorAttrDocs`

The `error` and `warning` function attributes can be used to specify a
custom diagnostic to be emitted when a call to such a function is not
eliminated via optimizations. This can be used to create compile time
assertions that depend on optimizations, while providing diagnostics
pointing to precise locations of the call site in the source.

```c++
__attribute__((warning("oh no"))) void dontcall();
void foo() {
  if (someCompileTimeAssertionThatsTrue)
    dontcall(); // Warning

  dontcall(); // Warning

  if (someCompileTimeAssertionThatsFalse)
    dontcall(); // No Warning
  sizeof(dontcall()); // No Warning
}
```

When the call occurs through inlined functions, the
`-fdiagnostics-show-inlining-chain` option can be used to show the
inlining chain that led to the call. This helps identify which call site
triggered the diagnostic when the attributed function is called from
multiple locations through inline functions.

When enabled, this option automatically uses debug info for accurate source
locations if available (`-gline-directives-only` (implicitly enabled at
`-g1`) or higher), or falls back to a heuristic based on metadata tracking.
When falling back, a note is emitted suggesting `-gline-directives-only` for
more accurate locations.


### exclude_from_explicit_instantiation

{clang-attr-syntaxes}`ExcludeFromExplicitInstantiationDocs`

The `exclude_from_explicit_instantiation` attribute opts-out a member of a
class template from being part of explicit template instantiations of that
class template. This means that an explicit instantiation will not instantiate
members of the class template marked with the attribute, but also that code
where an extern template declaration of the enclosing class template is visible
will not take for granted that an external instantiation of the class template
would provide those members (which would otherwise be a link error, since the
explicit instantiation won't provide those members). For example, let's say we
don't want the `data()` method to be part of libc++'s ABI. To make sure it
is not exported from the dylib, we give it hidden visibility:

```c++
// in <string>
template <class CharT>
class basic_string {
public:
  __attribute__((__visibility__("hidden")))
  const value_type* data() const noexcept { ... }
};

template class basic_string<char>;
```

Since an explicit template instantiation declaration for `basic_string<char>`
is provided, the compiler is free to assume that `basic_string<char>::data()`
will be provided by another translation unit, and it is free to produce an
external call to this function. However, since `data()` has hidden visibility
and the explicit template instantiation is provided in a shared library (as
opposed to simply another translation unit), `basic_string<char>::data()`
won't be found and a link error will ensue. This happens because the compiler
assumes that `basic_string<char>::data()` is part of the explicit template
instantiation declaration, when it really isn't. To tell the compiler that
`data()` is not part of the explicit template instantiation declaration, the
`exclude_from_explicit_instantiation` attribute can be used:

```c++
// in <string>
template <class CharT>
class basic_string {
public:
  __attribute__((__visibility__("hidden")))
  __attribute__((exclude_from_explicit_instantiation))
  const value_type* data() const noexcept { ... }
};

template class basic_string<char>;
```

Now, the compiler won't assume that `basic_string<char>::data()` is provided
externally despite there being an explicit template instantiation declaration:
the compiler will implicitly instantiate `basic_string<char>::data()` in the
TUs where it is used.

This attribute can be used on static and non-static member functions of class
templates, static data members of class templates and member classes of class
templates.

**Interaction with `__declspec(dllexport)` and `__declspec(dllimport)`**

For a DLL platform (i.e., Windows), this attribute also means "this member will
never be exported or imported". Despite its name, this semantics applies to
implicit instantiations and non-template entities as well.

```c++
// in <exception>
class __declspec(dllimport) nested_exception {
  ...
public:
  __attribute__((exclude_from_explicit_instantiation))
  exception_ptr nested_ptr() const noexcept { ... }
};
```

In this case, `nested_exception::nested_ptr` will never be attempted to be
imported.


### export_name, __funcref

{clang-attr-syntaxes}`WebAssemblyExportNameDocs`

Clang supports the `__attribute__((export_name(<name>)))`
attribute for the WebAssembly target. This attribute may be attached to a
function declaration, where it modifies how the symbol is to be exported
from the linked WebAssembly.

WebAssembly functions are exported via string name. By default when a symbol
is exported, the export name for C/C++ symbols are the same as their C/C++
symbol names. This attribute can be used to override the default behavior, and
request a specific string name be used instead.


### ext_vector_type

{clang-attr-syntaxes}`ExtVectorTypeDocs`

The `ext_vector_type(N)` attribute specifies that a type is a vector with N
elements, directly mapping to an LLVM vector type. Originally from OpenCL, it
allows element access the array subscript operator `[]`, `sN` where N is
a hexadecimal value, or `x, y, z, w` for graphics-style indexing.
This attribute enables efficient SIMD operations and is usable in
general-purpose code.

```c++
template <typename T, uint32_t N>
constexpr T simd_reduce(T [[clang::ext_vector_type(N)]] v) {
  static_assert((N & (N - 1)) == 0, "N must be a power of two");
  if constexpr (N == 1)
    return v[0];
  else
    return simd_reduce<T, N / 2>(v.hi + v.lo);
}
```

The vector type also supports swizzling up to sixteen elements. This can be done
using the object accessors. The OpenCL documentation lists all of the accepted
values.

```c++
using f16_x16 = _Float16 __attribute__((ext_vector_type(16)));

f16_x16 reverse(f16_x16 v) { return v.sfedcba9876543210; }
```

See the OpenCL documentation for some more complete examples.


### flatten

{clang-attr-syntaxes}`FlattenDocs`

The `flatten` attribute causes calls within the attributed function to
be inlined unless it is impossible to do so, for example if the body of the
callee is unavailable or if the callee has the `noinline` attribute.


### force_align_arg_pointer

{clang-attr-syntaxes}`X86ForceAlignArgPointerDocs`

Use this attribute to force stack alignment.

Legacy x86 code uses 4-byte stack alignment. Newer aligned SSE instructions
(like 'movaps') that work with the stack require operands to be 16-byte aligned.
This attribute realigns the stack in the function prologue to make sure the
stack can be used with SSE instructions.

Note that the x86_64 ABI forces 16-byte stack alignment at the call site.
Because of this, 'force_align_arg_pointer' is not needed on x86_64, except in
rare cases where the caller does not align the stack properly (e.g. flow
jumps from i386 arch code).

```c
__attribute__ ((force_align_arg_pointer))
void f () {
  ...
}
```


### format

{clang-attr-syntaxes}`FormatDocs`

Clang supports the `format` attribute, which indicates that the function
accepts (among other possibilities) a `printf` or `scanf`-like format string
and corresponding arguments or a `va_list` that contains these arguments.

Please see [GCC documentation about format attribute](http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html) to find details
about attribute syntax.

Clang implements two kinds of checks with this attribute.

1. Clang checks that the function with the `format` attribute is called with
   a format string that uses format specifiers that are allowed, and that
   arguments match the format string. This is the `-Wformat` warning, it is
   on by default.

2. Clang checks that the format string argument is a literal string. This is
   the `-Wformat-nonliteral` warning, it is off by default.

   Clang implements this mostly the same way as GCC, but there is a difference
   for functions that accept a `va_list` argument (for example, `vprintf`).
   GCC does not emit `-Wformat-nonliteral` warning for calls to such
   functions. Clang does not warn if the format string comes from a function
   parameter, where the function is annotated with a compatible attribute,
   otherwise it warns. For example:

   ```c
   __attribute__((__format__ (__scanf__, 1, 3)))
   void foo(const char* s, char *buf, ...) {
     va_list ap;
     va_start(ap, buf);

     vprintf(s, ap); // warning: format string is not a string literal
   }
   ```

   In this case we warn because `s` contains a format string for a
   `scanf`-like function, but it is passed to a `printf`-like function.

   If the attribute is removed, clang still warns, because the format string is
   not a string literal.

   Another example:

   ```c
   __attribute__((__format__ (__printf__, 1, 3)))
   void foo(const char* s, char *buf, ...) {
     va_list ap;
     va_start(ap, buf);

     vprintf(s, ap); // warning
   }
   ```

   In this case Clang does not warn because the format string `s` and
   the corresponding arguments are annotated. If the arguments are
   incorrect, the caller of `foo` will receive a warning.

As an extension to GCC's behavior, Clang accepts the `format` attribute on
non-variadic functions. Clang checks non-variadic format functions for the same
classes of issues that can be found on variadic functions, as controlled by the
same warning flags, except that the types of formatted arguments is forced by
the function signature. For example:

```c
__attribute__((__format__(__printf__, 1, 2)))
void fmt(const char *s, const char *a, int b);

void bar(void) {
  fmt("%s %i", "hello", 123); // OK
  fmt("%i %g", "hello", 123); // warning: arguments don't match format
  extern const char *fmt;
  fmt(fmt, "hello", 123); // warning: format string is not a string literal
}
```

When using the format attribute on a variadic function, the first data parameter
must be the index of the ellipsis in the parameter list. Clang will generate
a diagnostic otherwise, as it wouldn't be possible to forward that argument list
to `printf`-family functions. For instance, this is an error:

```c
__attribute__((__format__(__printf__, 1, 2)))
void fmt(const char *s, int b, ...);
// ^ error: format attribute parameter 3 is out of bounds
// (must be __printf__, 1, 3)
```

Using the `format` attribute on a non-variadic function emits a GCC
compatibility diagnostic.


### format_matches

{clang-attr-syntaxes}`FormatMatchesDocs`

The `format` attribute is the basis for the enforcement of diagnostics in the
`-Wformat` family, but it only handles the case where the format string is
passed along with the arguments it is going to format. It cannot handle the case
where the format string and the format arguments are passed separately from each
other. For instance:

```c
static const char *first_name;
static double todays_temperature;
static int wind_speed;

void say_hi(const char *fmt) {
  printf(fmt, first_name, todays_temperature);
      // ^ warning: format string is not a string literal
  printf(fmt, first_name, wind_speed);
      // ^ warning: format string is not a string literal
}

int main() {
  say_hi("hello %s, it is %g degrees outside");
  say_hi("hello %s, it is %d degrees outside!");
                        // ^ no diagnostic, but %d cannot format doubles
}
```

In this example, `fmt` is expected to format a `const char *` and a
`double`, but these values are not passed to `say_hi`. Without the
`format` attribute (which cannot apply in this case), the -Wformat-nonliteral
diagnostic unnecessarily triggers in the body of `say_hi`, and incorrect
`say_hi` call sites do not trigger a diagnostic.

To complement the `format` attribute, Clang also defines the
`format_matches` attribute. Its syntax is similar to the `format`
attribute's, but instead of taking the index of the first formatted value
argument, it takes a C string literal with the expected specifiers:

```c
static const char *first_name;
static double todays_temperature;
static int wind_speed;

__attribute__((__format_matches__(printf, 1, "%s %g")))
void say_hi(const char *fmt) {
  printf(fmt, first_name, todays_temperature); // no dignostic
  printf(fmt, first_name, wind_speed); // warning: format specifies type 'int' but the argument has type 'double'
}

int main() {
  say_hi("hello %s, it is %g degrees outside");
  say_hi("it is %g degrees outside, have a good day %s!");
  // warning: format specifies 'double' where 'const char *' is required
  // warning: format specifies 'const char *' where 'double' is required
}
```

The third argument to `format_matches` is expected to evaluate to a **C string
literal** even when the format string would normally be a different type for the
given flavor, like a `CFStringRef` or a `NSString *`.

The only requirement on the format string literal is that it has specifiers
that are compatible with the arguments that will be used. It can contain
arbitrary non-format characters. For instance, for the purposes of compile-time
validation, `"%s scored %g%% on her test"` and `"%s%g"` are interchangeable
as the format string argument. As a means of self-documentation, users may
prefer the former when it provides a useful example of an expected format
string.

In the implementation of a function with the `format_matches` attribute,
format verification works as if the format string was identical to the one
specified in the attribute.

```c
__attribute__((__format_matches__(printf, 1, "%s %g")))
void say_hi(const char *fmt) {
  printf(fmt, "person", 546);
                     // ^ warning: format specifies type 'double' but the
                     //   argument has type 'int'
  // note: format string is defined here:
  // __attribute__((__format_matches__(printf, 1, "%s %g")))
  //                                                  ^~
}
```

At the call sites of functions with the `format_matches` attribute, format
verification instead compares the two format strings to evaluate their
equivalence. Each format flavor defines equivalence between format specifiers.
Generally speaking, two specifiers are equivalent if they format the same type.
For instance, in the `printf` flavor, `%2i` and `%-0.5d` are compatible.
When `-Wformat-signedness` is disabled, `%d` and `%u` are compatible. For
a negative example, `%ld` is incompatible with `%d`.

Do note the following un-obvious cases:

- Passing `NULL` as the format string does not trigger format diagnostics.
- When the format string is not NULL, it cannot miss specifiers, even in
  trailing positions. For instance, `%d` is not accepted when the required
  format is `%d %d %d`.
- While checks for the `format` attribute tolerate sone size mismatches
  that standard argument promotion renders immaterial (such as formatting an
  `int` with `%hhd`, which specifies a `char`-sized integer), checks for
  `format_matches` require specified argument sizes to match exactly.
- Format strings expecting a variable modifier (such as `%*s`) are
  incompatible with format strings that would itemize the variable modifiers
  (such as `%i %s`), even if the two specify ABI-compatible argument lists.
- All pointer specifiers, modifiers aside, are mutually incompatible. For
  instance, `%s` is not compatible with `%p`, and `%p` is not compatible
  with `%n`, and `%hhn` is incompatible with `%s`, even if the pointers
  are ABI-compatible or identical on the selected platform. However, `%0.5s`
  is compatible with `%s`, since the difference only exists in modifier flags.
  This is not overridable with `-Wformat-pedantic` or its inverse, which
  control similar behavior in `-Wformat`.

At this time, clang implements `format_matches` only for format types in the
`printf` family. This includes variants such as Apple's NSString format and
the FreeBSD `kprintf`, but excludes `scanf`. Using a known but unsupported
format silently fails in order to be compatible with other implementations that
would support these formats.


### function_return

{clang-attr-syntaxes}`FunctionReturnThunksDocs`

The attribute `function_return` can replace return instructions with jumps to
target-specific symbols. This attribute supports 2 possible values,
corresponding to the values supported by the `-mfunction-return=` command
line flag:

- `__attribute__((function_return("keep")))` to disable related transforms.
  This is useful for undoing global setting from `-mfunction-return=` locally
  for individual functions.
- `__attribute__((function_return("thunk-extern")))` to replace returns with
  jumps, while NOT emitting the thunk.

The values `thunk` and `thunk-inline` from GCC are not supported.

The symbol used for `thunk-extern` is target specific:

- X86: `__x86_return_thunk`

As such, this function attribute is currently only supported on X86 targets.


### gnu_inline

{clang-attr-syntaxes}`GnuInlineDocs`

The `gnu_inline` changes the meaning of `extern inline` to use GNU inline
semantics, meaning:

- If any declaration that is declared `inline` is not declared `extern`,
  then the `inline` keyword is just a hint. In particular, an out-of-line
  definition is still emitted for a function with external linkage, even if all
  call sites are inlined, unlike in C99 and C++ inline semantics.
- If all declarations that are declared `inline` are also declared
  `extern`, then the function body is present only for inlining and no
  out-of-line version is emitted.

Some important consequences: `static inline` emits an out-of-line
version if needed, a plain `inline` definition emits an out-of-line version
always, and an `extern inline` definition (in a header) followed by a
(non-`extern`) `inline` declaration in a source file emits an out-of-line
version of the function in that source file but provides the function body for
inlining to all includers of the header.

Either `__GNUC_GNU_INLINE__` (GNU inline semantics) or
`__GNUC_STDC_INLINE__` (C99 semantics) will be defined (they are mutually
exclusive). If `__GNUC_STDC_INLINE__` is defined, then the `gnu_inline`
function attribute can be used to get GNU inline semantics on a per function
basis. If `__GNUC_GNU_INLINE__` is defined, then the translation unit is
already being compiled with GNU inline semantics as the implied default. It is
unspecified which macro is defined in a C++ compilation.

GNU inline semantics are the default behavior with `-std=gnu89`,
`-std=c89`, `-fgnu89-inline`, or `-std=iso9899:199409`.


### guard

{clang-attr-syntaxes}`CFGuardDocs`

Code can indicate CFG checks are not wanted with the `__declspec(guard(nocf))`
attribute. This directs the compiler to not insert any CFG checks for the entire
function. This approach is typically used only sparingly in specific situations
where the programmer has manually inserted "CFG-equivalent" protection. The
programmer knows that they are calling through some read-only function table
whose address is obtained through read-only memory references and for which the
index is masked to the function table limit. This approach may also be applied
to small wrapper functions that are not inlined and that do nothing more than
make a call through a function pointer. Since incorrect usage of this directive
can compromise the security of CFG, the programmer must be very careful using
the directive. Typically, this usage is limited to very small functions that
only call one function.

Control Flow Guard documentation is available here:
<https://docs.microsoft.com/en-us/windows/win32/secbp/pe-metadata>


### hot

{clang-attr-syntaxes}`HotFunctionEntryDocs`

`__attribute__((hot))` marks a function as hot, as a manual alternative to PGO hotness data.
If PGO data is available, the annotation `__attribute__((hot))` overrides the profile count based hotness (unlike `__attribute__((cold))`).


### hybrid_patchable

{clang-attr-syntaxes}`HybridPatchableDocs`

The `hybrid_patchable` attribute declares an ARM64EC function with an additional
x86-64 thunk, which may be patched at runtime.

For more information see
[ARM64EC ABI documentation](https://learn.microsoft.com/en-us/windows/arm/arm64ec-abi).


### ifunc

{clang-attr-syntaxes}`IFuncDocs`

`__attribute__((ifunc("resolver")))` is used to mark that the address of a
declaration should be resolved at runtime by calling a resolver function.

The symbol name of the resolver function is given in quotes. A function with
this name (after mangling) must be defined in the current translation unit; it
may be `static`. The resolver function should return a pointer.

The `ifunc` attribute may only be used on a function declaration. A function
declaration with an `ifunc` attribute is considered to be a definition of the
declared entity. The entity must not have weak linkage; for example, in C++,
it cannot be applied to a declaration if a definition at that location would be
considered inline.

Not all targets support this attribute:

- ELF target support depends on both the linker and runtime linker, and is
  available in at least lld 4.0 and later, binutils 2.20.1 and later, glibc
  v2.11.1 and later, and FreeBSD 9.1 and later.
- Mach-O targets support it, but with slightly different semantics: the resolver
  is run at first call, instead of at load time by the runtime linker.
- Windows target supports it on AArch64, but with different semantics: the
  `ifunc` is replaced with a global function pointer, and the call is replaced
  with an indirect call. The function pointer is initialized by a constructor
  that calls the resolver.
- Baremetal target supports it on AVR.
- AIX/XCOFF supports it via a compiler-only solution. An ifunc appears as a
  regular function (has an entry point `.foo[PR]` and a function descriptor
  `foo[DS]`). The entry point is a stub that branches to the function address
  in the descriptor, and the descriptor is initialized via a constructor
  function (`__init_ifuncs`) that is linked into every shared object and
  executable. `__init_ifuncs` calls the resolver of each ifunc and stores the
  result in the corresponding descriptor.
- Other targets currently do not support this attribute.


### import_module

{clang-attr-syntaxes}`WebAssemblyImportModuleDocs`

Clang supports the `__attribute__((import_module(<module_name>)))`
attribute for the WebAssembly target. This attribute may be attached to a
function declaration, where it modifies how the symbol is to be imported
within the WebAssembly linking environment.

WebAssembly imports use a two-level namespace scheme, consisting of a module
name, which typically identifies a module from which to import, and a field
name, which typically identifies a field from that module to import. By
default, module names for C/C++ symbols are assigned automatically by the
linker. This attribute can be used to override the default behavior, and
request a specific module name be used instead.


### import_name

{clang-attr-syntaxes}`WebAssemblyImportNameDocs`

Clang supports the `__attribute__((import_name(<name>)))`
attribute for the WebAssembly target. This attribute may be attached to a
function declaration, where it modifies how the symbol is to be imported
within the WebAssembly linking environment.

WebAssembly imports use a two-level namespace scheme, consisting of a module
name, which typically identifies a module from which to import, and a field
name, which typically identifies a field from that module to import. By
default, field names for C/C++ symbols are the same as their C/C++ symbol
names. This attribute can be used to override the default behavior, and
request a specific field name be used instead.


### internal_linkage

{clang-attr-syntaxes}`InternalLinkageDocs`

The `internal_linkage` attribute changes the linkage type of the declaration
to internal. This is similar to C-style `static`, but can be used on classes
and class methods. When applied to a class definition, this attribute affects
all methods and static data members of that class. This can be used to contain
the ABI of a C++ library by excluding unwanted class methods from the export
tables.


### interrupt (ARM)

{clang-attr-syntaxes}`ARMInterruptDocs`

Clang supports the GNU style `__attribute__((interrupt("TYPE")))` attribute on
ARM targets. This attribute may be attached to a function definition and
instructs the backend to generate appropriate function entry/exit code so that
it can be used directly as an interrupt service routine.

The parameter passed to the interrupt attribute is optional, but if
provided it must be a string literal with one of the following values: "IRQ",
"FIQ", "SWI", "ABORT", "UNDEF".

The semantics are as follows:

- If the function is AAPCS, Clang instructs the backend to realign the stack to
  8 bytes on entry. This is a general requirement of the AAPCS at public
  interfaces, but may not hold when an exception is taken. Doing this allows
  other AAPCS functions to be called.

- If the CPU is M-class this is all that needs to be done since the architecture
  itself is designed in such a way that functions obeying the normal AAPCS ABI
  constraints are valid exception handlers.

- If the CPU is not M-class, the prologue and epilogue are modified to save all
  non-banked registers that are used, so that upon return the user-mode state
  will not be corrupted. Note that to avoid unnecessary overhead, only
  general-purpose (integer) registers are saved in this way. If VFP operations
  are needed, that state must be saved manually.

  Specifically, interrupt kinds other than "FIQ" will save all core registers
  except "lr" and "sp". "FIQ" interrupts will save r0-r7.

- If the CPU is not M-class, the return instruction is changed to one of the
  canonical sequences permitted by the architecture for exception return. Where
  possible the function itself will make the necessary "lr" adjustments so that
  the "preferred return address" is selected.

  Unfortunately the compiler is unable to make this guarantee for an "UNDEF"
  handler, where the offset from "lr" to the preferred return address depends on
  the execution state of the code which generated the exception. In this case
  a sequence equivalent to "movs pc, lr" will be used.


### interrupt (AVR)

{clang-attr-syntaxes}`AVRInterruptDocs`

Clang supports the GNU style `__attribute__((interrupt))` attribute on
AVR targets. This attribute may be attached to a function definition and instructs
the backend to generate appropriate function entry/exit code so that it can be used
directly as an interrupt service routine.

On the AVR, the hardware globally disables interrupts when an interrupt is executed.
The first instruction of an interrupt handler declared with this attribute is a SEI
instruction to re-enable interrupts. See also the signal attribute that
does not insert a SEI instruction.


### interrupt (MIPS)

{clang-attr-syntaxes}`MipsInterruptDocs`

Clang supports the GNU style `__attribute__((interrupt("ARGUMENT")))` attribute on
MIPS targets. This attribute may be attached to a function definition and instructs
the backend to generate appropriate function entry/exit code so that it can be used
directly as an interrupt service routine.

By default, the compiler will produce a function prologue and epilogue suitable for
an interrupt service routine that handles an External Interrupt Controller (eic)
generated interrupt. This behavior can be explicitly requested with the "eic"
argument.

Otherwise, for use with vectored interrupt mode, the argument passed should be
of the form "vector=LEVEL" where LEVEL is one of the following values:
"sw0", "sw1", "hw0", "hw1", "hw2", "hw3", "hw4", "hw5". The compiler will
then set the interrupt mask to the corresponding level which will mask all
interrupts up to and including the argument.

The semantics are as follows:

- The prologue is modified so that the Exception Program Counter (EPC) and
  Status coprocessor registers are saved to the stack. The interrupt mask is
  set so that the function can only be interrupted by a higher priority
  interrupt. The epilogue will restore the previous values of EPC and Status.
- The prologue and epilogue are modified to save and restore all non-kernel
  registers as necessary.
- The FPU is disabled in the prologue, as the floating pointer registers are not
  spilled to the stack.
- The function return sequence is changed to use an exception return instruction.
- The parameter sets the interrupt mask for the function corresponding to the
  interrupt level specified. If no mask is specified the interrupt mask
  defaults to "eic".


### interrupt (RISC-V)

{clang-attr-syntaxes}`RISCVInterruptDocs`

Clang supports the GNU style `__attribute__((interrupt))` attribute on RISCV
targets. This attribute may be attached to a function definition and instructs
the backend to generate appropriate function entry/exit code so that it can be
used directly as an interrupt service routine.

Permissible values for this parameter are `machine`, `supervisor`,
`rnmi`, `qci-nest`, `qci-nonest`, `SiFive-CLIC-preemptible`, and
`SiFive-CLIC-stack-swap`. If there is no parameter, then it defaults to
`machine`.

The `rnmi` value is used for resumable non-maskable interrupts. It requires the
standard Smrnmi extension.

The `qci-nest` and `qci-nonest` values require Qualcomm's Xqciint extension
and are used for Machine-mode Interrupts and Machine-mode Non-maskable
interrupts. These use the following instructions from Xqciint to save and
restore interrupt state to the stack -- the `qci-nest` value will use
`qc.c.mienter.nest` and the `qci-nonest` value will use `qc.c.mienter` to
begin the interrupt handler. Both of these will use `qc.c.mileaveret` to
restore the state and return to the previous context.

The `SiFive-CLIC-preemptible` and `SiFive-CLIC-stack-swap` values are used
for machine-mode interrupts. For `SiFive-CLIC-preemptible` interrupts, the
values of `mcause` and `mepc` are saved onto the stack, and interrupts are
re-enabled. For `SiFive-CLIC-stack-swap` interrupts, the stack pointer is
swapped with `mscratch` before its first use and after its last use.

The SiFive CLIC values may be combined with each other and with the `machine`
attribute value. Any other combination of different values is not allowed.

Repeated interrupt attribute on the same declaration will cause a warning
to be emitted. In case of repeated declarations, the last one prevails.

References:
- [GCC RISC-V Attributes](https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Function-Attributes.html)
- [The RISC-V Instruction Set Manual Volume II: Privileged Architecture Version 1.10](https://docs.riscv.org/reference/isa/v1.10/_attachments/riscv-privileged.pdf)
- [Xqci extension v0.13](https://github.com/quic/riscv-unified-db/releases/tag/Xqci-0.13.0)
- [SiFive Interrupt Cookbook Version 1.2](https://sifive.cdn.prismic.io/sifive/d1984d2b-c9b9-4c91-8de0-d68a5e64fa0f_sifive-interrupt-cookbook-v1p2.pdf)


### interrupt (X86)

{clang-attr-syntaxes}`AnyX86InterruptDocs`

Clang supports the GNU style `__attribute__((interrupt))` attribute on X86
targets. This attribute may be attached to a function definition and instructs
the backend to generate appropriate function entry/exit code so that it can be
used directly as an interrupt service routine.

Interrupt handlers have access to the stack frame pushed onto the stack by the processor,
and return using the `IRET` instruction. All registers in an interrupt handler are callee-saved.
Exception handlers also have access to the error code pushed onto the stack by the processor,
when applicable.

An interrupt handler must take the following arguments:

```c
__attribute__ ((interrupt))
void f (struct stack_frame *frame) {
    ...
}
```

Where `struct stack_frame` is a suitable struct matching the stack frame pushed
by the processor.

An exception handler must take the following arguments:

```c
__attribute__ ((interrupt))
void g (struct stack_frame *frame, unsigned long code) {
    ...
}
```

On 32-bit targets, the `code` argument should be of type `unsigned int`.

Exception handlers should only be used when an error code is pushed by the processor.
Using the incorrect handler type will crash the system.

Interrupt and exception handlers cannot be called by other functions and must have return type `void`.

Interrupt and exception handlers should only call functions with the `no_caller_saved_registers`
attribute, or should be compiled with the `-mgeneral-regs-only` flag to avoid saving unused
non-GPR registers.


### interrupt_save_fp (ARM)

{clang-attr-syntaxes}`ARMInterruptSaveFPDocs`

Clang supports the GNU style `__attribute__((interrupt_save_fp("TYPE")))`
on ARM targets. This attribute behaves the same way as the ARM interrupt
attribute, except the general purpose floating point registers are also saved,
along with FPEXC and FPSCR. Note, even on M-class CPUs, where the floating
point context can be automatically saved depending on the FPCCR, the general
purpose floating point registers will be saved.


### launch_bounds

{clang-attr-syntaxes}`LaunchBoundsDocs`

The `__launch_bounds__` attribute (also spelled `launch_bounds`) originates
in CUDA. It informs the compiler of the launch configuration a kernel will be
dispatched with, allowing it to optimize the kernel accordingly. It takes the
form `__launch_bounds__(<max-threads-per-block>[,
<min-blocks-per-multiprocessor>[, <max-blocks-per-cluster>]])`. All arguments
are constant expressions.

The attribute only takes effect on `__global__` (kernel) functions; like
NVCC, Clang ignores it on any other function.

`<max-threads-per-block>` specifies the maximum number of threads per block
the kernel will be launched with. `<min-blocks-per-multiprocessor>` specifies
the desired minimum number of blocks resident per multiprocessor, and
`<max-blocks-per-cluster>` the maximum number of blocks per cluster.

For the NVPTX target, `<max-threads-per-block>` and
`<min-blocks-per-multiprocessor>` map to the `.maxntid` and `.minnctapersm`
PTX directives, respectively, and `<max-blocks-per-cluster>` (which requires
`sm_90` or newer) maps to `.maxclusterrank`.

For the AMDGPU target, the attribute is translated into the equivalent AMDGPU
kernel attributes:

  - `<max-threads-per-block>` sets the maximum
    `amdgpu_flat_work_group_size` (as `1, <max-threads-per-block>`).
  - `<min-blocks-per-multiprocessor>` sets the minimum
    `amdgpu_waves_per_eu`. Note that HIP reinterprets this CUDA argument as a
    minimum number of waves per execution unit, so its meaning differs from the
    NVPTX interpretation.
  - `<max-blocks-per-cluster>` is currently ignored.

An explicit `amdgpu_flat_work_group_size` or `amdgpu_waves_per_eu` attribute
takes precedence over the value derived from `__launch_bounds__`.

When the same kernel is declared multiple times, the launch bounds from the most
recent declaration that specifies them are used; a definition without
`__launch_bounds__` inherits the bounds from an earlier declaration.


### lifetime_capture_by, lifetime_capture_by_this, lifetime_capture_by_global, lifetime_capture_by_unknown

{clang-attr-syntaxes}`LifetimeCaptureByDocs`

Similar to [lifetimebound], the `lifetime_capture_by` attribute family on a
function parameter or implicit object parameter indicates that a capturing
entity may refer to the object referred to by that parameter. The capturing
entity can be named in `lifetime_capture_by(X)` or selected by one of the
standalone special forms listed below.

Below is a list of types of the parameters and what they're considered to refer to:

- A reference param (of non-view type) is considered to refer to its referenced object.
- A pointer param (of non-view type) is considered to refer to its pointee.
- View type param (type annotated with `[[gsl::Pointer()]]`) is considered to refer
  to its pointee (gsl owner). This holds true even if the view type appears as a reference
  in the parameter. For example, both `std::string_view` and
  `const std::string_view &` are considered to refer to a `std::string`.
- A `std::initializer_list<T>` is considered to refer to its underlying array.
- Aggregates (arrays and simple `struct`s) are considered to refer to all
  objects that their transitive subobjects refer to.

Clang would diagnose when a temporary object is used as an argument to such an
annotated parameter.
In this case, the capturing entity `X` could capture a dangling reference to this
temporary object.

```c++
void addToSet(std::string_view a [[clang::lifetime_capture_by(s)]], std::set<std::string_view>& s) {
  s.insert(a);
}
void use() {
  std::set<std::string_view> s;
  addToSet(std::string(), s); // Warning: object whose reference is captured by 's' will be destroyed at the end of the full-expression.
  //       ^^^^^^^^^^^^^
  std::string local;
  addToSet(local, s); // Ok.
}
```

The capturing entity can be one of the following:

- Another (named) function parameter.

  ```c++
  void addToSet(std::string_view a [[clang::lifetime_capture_by(s)]], std::set<std::string_view>& s) {
    s.insert(a);
  }
  ```

- `this` (in case of member functions), written as
  `lifetime_capture_by_this`.

  ```c++
  class S {
    void addToSet(std::string_view a [[clang::lifetime_capture_by_this]]) {
      s.insert(a);
    }
    std::set<std::string_view> s;
  };
  ```

  Note: When applied to a constructor parameter, `[[clang::lifetime_capture_by_this]]` is just an alias of `[[clang::lifetimebound]]`.

- `global` and `unknown`, written as `lifetime_capture_by_global` and
  `lifetime_capture_by_unknown` respectively.

  ```c++
  std::set<std::string_view> s;
  void addToSet(std::string_view a [[clang::lifetime_capture_by_global]]) {
    s.insert(a);
  }
  void addSomewhere(std::string_view a [[clang::lifetime_capture_by_unknown]]);
  ```

The attribute can be applied to the implicit `this` parameter of a member
function by writing the attribute after the function type:

```c++
struct S {
  const char *data(std::set<S*>& s) [[clang::lifetime_capture_by(s)]] {
    s.insert(this);
  }
};
```

The parameter-list form supports specifying more than one capturing entity:

```c++
void addToSets(std::string_view a [[clang::lifetime_capture_by(s1, s2)]],
               std::set<std::string_view>& s1,
               std::set<std::string_view>& s2) {
  s1.insert(a);
  s2.insert(a);
}
```

Distinct `lifetime_capture_by` forms can also be combined on the same
declaration, but each form can appear at most once. For example,
`[[clang::lifetime_capture_by(s), clang::lifetime_capture_by_this]]` is
allowed, but two `[[clang::lifetime_capture_by(...)]]` attributes or two
`[[clang::lifetime_capture_by_this]]` attributes on the same declaration are
rejected.

Limitation: The capturing entity `X` is not used by the analysis and is
used for documentation purposes only. This is because the analysis is
statement-local and only detects use of a temporary as an argument to the
annotated parameter.

```c++
void addToSet(std::string_view a [[clang::lifetime_capture_by(s)]], std::set<std::string_view>& s);
void use() {
  std::set<std::string_view> s;
  if (foo()) {
    std::string str;
    addToSet(str, s); // Not detected.
  }
}
```


### lifetimebound

{clang-attr-syntaxes}`LifetimeBoundDocs`

The `lifetimebound` attribute on a function parameter or implicit object
parameter indicates that objects that are referred to by that parameter may
also be referred to by the return value of the annotated function (or, for a
parameter of a constructor, by the value of the constructed object).

By default, a reference is considered to refer to its referenced object, a
pointer is considered to refer to its pointee, a `std::initializer_list<T>`
is considered to refer to its underlying array, and aggregates (arrays and
simple `struct`s) are considered to refer to all objects that their
transitive subobjects refer to.

Clang warns if it is able to detect that an object or reference refers to
another object with a shorter lifetime. For example, Clang will warn if a
function returns a reference to a local variable, or if a reference is bound to
a temporary object whose lifetime is not extended. By using the
`lifetimebound` attribute, this determination can be extended to look through
user-declared functions. For example:

```c++
#include <map>
#include <string>

using namespace std::literals;

// Returns m[key] if key is present, or default_value if not.
template<typename T, typename U>
const U &get_or_default(const std::map<T, U> &m [[clang::lifetimebound]],
                        const T &key, /* note, not lifetimebound */
                        const U &default_value [[clang::lifetimebound]]) {
  if (auto iter = m.find(key); iter != m.end()) return iter->second;
  else return default_value;
}

int main() {
  std::map<std::string, std::string> m;
  // warning: temporary bound to local reference 'val1' will be destroyed
  // at the end of the full-expression
  const std::string &val1 = get_or_default(m, "foo"s, "bar"s);

  // No warning in this case.
  std::string def_val = "bar"s;
  const std::string &val2 = get_or_default(m, "foo"s, def_val);

  return 0;
}
```

The attribute can be applied to the implicit `this` parameter of a member
function by writing the attribute after the function type:

```c++
struct string {
  // The returned pointer should not outlive '*this'.
  const char *data() const [[clang::lifetimebound]];
};
```

This attribute is inspired by the C++ committee paper [P0936R0](http://wg21.link/p0936r0), but does not affect whether temporary objects
have their lifetimes extended.


### long_call, far

{clang-attr-syntaxes}`MipsLongCallStyleDocs`

Clang supports the `__attribute__((long_call))`, `__attribute__((far))`,
and `__attribute__((near))` attributes on MIPS targets. These attributes may
only be added to function declarations and change the code generated
by the compiler when directly calling the function. The `near` attribute
allows calls to the function to be made using the `jal` instruction, which
requires the function to be located in the same naturally aligned 256MB
segment as the caller. The `long_call` and `far` attributes are synonyms
and require the use of a different call sequence that works regardless
of the distance between the functions.

These attributes have no effect for position-independent code.

These attributes take priority over command line switches such
as `-mlong-calls` and `-mno-long-calls`.


### malloc

{clang-attr-syntaxes}`RestrictDocs`

The `malloc` attribute has two forms with different functionality. The first
is when it is used without arguments, where it marks that a function acts like
a system memory allocation function, returning a pointer to allocated storage
that does not alias storage from any other object accessible to the caller.

The second form is when `malloc` takes one or two arguments. The first
argument names a function that should be associated with this function as its
deallocation function. When this form is used, it enables the compiler to
diagnose when the incorrect deallocation function is used with this variable.
However the associated warning, spelled `-Wmismatched-dealloc` in GCC, is not
yet implemented in clang.


### malloc_span

{clang-attr-syntaxes}`MallocSpanDocs`

The `malloc_span` attribute can be used to mark that a function which acts
like a system memory allocation function and returns a span-like structure,
where the returned memory range does not alias storage from any other object
accessible to the caller.

In this context, a span-like structure is assumed to have two non-static data
members, one of which is a pointer to the start of the allocated memory and
the other one is either an integer type containing the size of the actually
allocated memory or a pointer to the end of the allocated region. Note, static
data members do not impact whether a type is span-like or not.

In combination with the `alloc_size` attribute, if the begin pointer is
non-null, the size of the returned span-like object has to be greater or equal
to the number of bytes guaranteed to be dereferenceable by `alloc_size`. It also
guarantees that the number of dereferenceable bytes is at least size.


### micromips, nomicromips

{clang-attr-syntaxes}`MicroMipsDocs`

Clang supports the GNU style `__attribute__((micromips))` and
`__attribute__((nomicromips))` attributes on MIPS targets. These attributes
may be attached to a function definition and instructs the backend to generate
or not to generate microMIPS code for that function.

These attributes override the `-mmicromips` and `-mno-micromips` options
on the command line.


### mig_server_routine

{clang-attr-syntaxes}`MIGConventionDocs`

The Mach Interface Generator release-on-success convention dictates

functions that follow it to only release arguments passed to them when they
return "success" (a `kern_return_t` error code that indicates that
no errors have occurred). Otherwise the release is performed by the MIG client
that called the function. The annotation `__attribute__((mig_server_routine))`
is applied in order to specify which functions are expected to follow the
convention. This allows the Static Analyzer to find bugs caused by violations of
that convention. The attribute would normally appear on the forward declaration
of the actual server routine in the MIG server header, but it may also be
added to arbitrary functions that need to follow the same convention - for
example, a user can add them to auxiliary functions called by the server routine
that have their return value of type `kern_return_t` unconditionally returned
from the routine. The attribute can be applied to C++ methods, and in this case
it will be automatically applied to overrides if the method is virtual. The
attribute can also be written using C++11 syntax: `[[mig::server_routine]]`.


### min_vector_width

{clang-attr-syntaxes}`MinVectorWidthDocs`

Clang supports the `__attribute__((min_vector_width(width)))` attribute. This
attribute may be attached to a function and informs the backend that this
function desires vectors of at least this width to be generated. Target-specific
maximum vector widths still apply. This means even if you ask for something
larger than the target supports, you will only get what the target supports.
This attribute is meant to be a hint to control target heuristics that may
generate narrower vectors than what the target hardware supports.

This is currently used by the X86 target to allow some CPUs that support 512-bit
vectors to be limited to using 256-bit vectors to avoid frequency penalties.
This is currently enabled with the `-prefer-vector-width=256` command line
option. The `min_vector_width` attribute can be used to prevent the backend
from trying to split vector operations to match the `prefer-vector-width`. All
X86 vector intrinsics from x86intrin.h already set this attribute. Additionally,
use of any of the X86-specific vector builtins will implicitly set this
attribute on the calling function. The intent is that explicitly writing vector
code using the X86 intrinsics will prevent `prefer-vector-width` from
affecting the code.


### minsize

{clang-attr-syntaxes}`MinSizeDocs`

This function attribute indicates that optimization passes and code generator passes
make choices that keep the function code size as small as possible. Optimizations may
also sacrifice runtime performance in order to minimize the size of the generated code.


### modular_format

{clang-attr-syntaxes}`ModularFormatDocs`

The `modular_format` attribute can be applied to a function that bears the
`format` attribute (or standard library functions) to indicate that the
implementation is "modular", that is, that the implementation is logically
divided into a number of named aspects. When the compiler can determine that
not all aspects of the implementation are needed for a given call, the compiler
may redirect the call to the identifier given as the first argument to the
attribute (the modular implementation function).

The second argument is an implementation name, and the remaining arguments are
aspects of the format string for the compiler to report. The implementation
name is an unevaluated identifier in the C namespace.

The compiler reports that a call requires an aspect by issuing a relocation for
the symbol `<impl_name>_<aspect>` at the point of the call. This arranges for
code and data needed to support the aspect of the implementation to be brought
into the link to satisfy weak references in the modular implemenation function.
If the compiler does not understand an aspect, it must summarily consider any
call to require that aspect.

For example, say `printf` is annotated with
`modular_format(__modular_printf, "__printf", "float")`. Then, a call to
`printf(var, 42)` would be untouched. A call to `printf("%d", 42)` would
become a call to `__modular_printf` with the same arguments, as would
`printf("%f", 42.0)`. The latter would be accompanied with a strong
relocation against the symbol `__printf_float`, which would bring floating
point support for `printf` into the link.

If the attribute appears more than once on a declaration, or across a chain of
redeclarations, it is an error for the attributes to have different arguments,
excepting that the aspects may be in any order.

The following aspects are currently supported:

- `fixed`: The call has a C ISO 18037 fixed-point argument.
- `float`: The call has a floating-point argument.


### no_builtin

{clang-attr-syntaxes}`NoBuiltinDocs`

The `__attribute__((no_builtin))` is similar to the `-fno-builtin` flag
except it is specific to the body of a function. The attribute may also be
applied to a virtual function but has no effect on the behavior of overriding
functions in a derived class.

It accepts one or more strings corresponding to the specific names of the
builtins to disable (e.g. "memcpy", "memset").
If the attribute is used without parameters it will disable all buitins at
once.

```c++
// The compiler is not allowed to add any builtin to foo's body.
void foo(char* data, size_t count) __attribute__((no_builtin)) {
  // The compiler is not allowed to convert the loop into
  // `__builtin_memset(data, 0xFE, count);`.
  for (size_t i = 0; i < count; ++i)
    data[i] = 0xFE;
}

// The compiler is not allowed to add the `memcpy` builtin to bar's body.
void bar(char* data, size_t count) __attribute__((no_builtin("memcpy"))) {
  // The compiler is allowed to convert the loop into
  // `__builtin_memset(data, 0xFE, count);` but cannot generate any
  // `__builtin_memcpy`
  for (size_t i = 0; i < count; ++i)
    data[i] = 0xFE;
}
```


### no_caller_saved_registers

{clang-attr-syntaxes}`AnyX86NoCallerSavedRegistersDocs`

Use this attribute to indicate that the specified function has no
caller-saved registers. That is, all registers are callee-saved except for
registers used for passing parameters to the function or returning parameters
from the function.
The compiler saves and restores any modified registers that were not used for
passing or returning arguments to the function.

The user can call functions specified with the `no_caller_saved_registers`
attribute from an interrupt handler without saving and restoring all
call-clobbered registers.

Functions specified with the `no_caller_saved_registers` attribute should only
call other functions with the `no_caller_saved_registers` attribute, or should be
compiled with the `-mgeneral-regs-only` flag to avoid saving unused non-GPR registers.

Note that `no_caller_saved_registers` attribute is not a calling convention.
In fact, it only overrides the decision of which registers should be saved by
the caller, but not how the parameters are passed from the caller to the callee.

For example:

```c
__attribute__ ((no_caller_saved_registers, fastcall))
void f (int arg1, int arg2) {
  ...
}
```

In this case parameters `arg1` and `arg2` will be passed in registers.
In this case, on 32-bit x86 targets, the function `f` will use ECX and EDX as
register parameters. However, it will not assume any scratch registers and
should save and restore any modified registers except for ECX and EDX.


### no_outline

{clang-attr-syntaxes}`NoOutlineDocs`

This function attribute suppresses outlining from the annotated function.

Outlining is the process where common parts of separate functions are extracted
into a separate function (or assembly snippet), and calls to that function or
snippet are inserted in the original functions. In this way, it can be seen as
the opposite of inlining. It can help to reduce code size.


### no_profile_instrument_function

{clang-attr-syntaxes}`NoProfileInstrumentFunctionDocs`

Use the `no_profile_instrument_function` attribute on a function declaration
to denote that the compiler should not instrument the function with
profile-related instrumentation, such as via the
`-fprofile-generate` / `-fprofile-instr-generate` /
`-fcs-profile-generate` / `-fprofile-arcs` flags.


### no_sanitize

{clang-attr-syntaxes}`NoSanitizeDocs`

Use the `no_sanitize` attribute on a function or a global variable
declaration to specify that a particular instrumentation or set of
instrumentations should not be applied.

The attribute takes a list of string literals with the following accepted
values:

- all values accepted by `-fno-sanitize=`;
- `coverage`, to disable SanitizerCoverage instrumentation.

For example, `__attribute__((no_sanitize("address", "thread")))` specifies
that AddressSanitizer and ThreadSanitizer should not be applied to the function
or variable. Using `__attribute__((no_sanitize("coverage")))` specifies that
SanitizerCoverage should not be applied to the function.

See {ref}`Controlling Code Generation <controlling-code-generation>` for a
full list of supported sanitizer flags.


(langext-address_sanitizer)=

### no_sanitize_address, no_address_safety_analysis

{clang-attr-syntaxes}`NoSanitizeAddressDocs`

Use `__attribute__((no_sanitize_address))` on a function or a global
variable declaration to specify that address safety instrumentation
(e.g. AddressSanitizer) should not be applied.


(langext-memory_sanitizer)=

### no_sanitize_memory

{clang-attr-syntaxes}`NoSanitizeMemoryDocs`

Use `__attribute__((no_sanitize_memory))` on a function declaration to
specify that checks for uninitialized memory should not be inserted
(e.g. by MemorySanitizer). The function may still be instrumented by the tool
to avoid false positives in other places.


(langext-thread_sanitizer)=

### no_sanitize_thread

{clang-attr-syntaxes}`NoSanitizeThreadDocs`

Use `__attribute__((no_sanitize_thread))` on a function declaration to
specify that checks for data races on plain (non-atomic) memory accesses should
not be inserted by ThreadSanitizer. The function is still instrumented by the
tool to avoid false positives and provide meaningful stack traces.


### no_speculative_load_hardening

{clang-attr-syntaxes}`NoSpeculativeLoadHardeningDocs`

This attribute can be applied to a function declaration in order to indicate
that [Speculative Load Hardening][slh] is *not* needed for the function body.
This can also be applied to a method in Objective C. This attribute will take
precedence over the command line flag in the case where
{option}`-mspeculative-load-hardening` is specified.

Warning: This attribute may not prevent Speculative Load Hardening from being
enabled for a function which inlines a function that has the
`speculative_load_hardening` attribute. This is intended to provide a
maximally conservative model where the code that is marked with the
`speculative_load_hardening` attribute will always (even when inlined)
be hardened. A user of this attribute may want to mark functions called by
a function they do not want to be hardened with the `noinline` attribute.

For example:

```c
__attribute__((speculative_load_hardening))
int foo(int i) {
  return i;
}

// Note: bar() may still have speculative load hardening enabled if
// foo() is inlined into bar(). Mark foo() with __attribute__((noinline))
// to avoid this situation.
__attribute__((no_speculative_load_hardening))
int bar(int i) {
  return foo(i);
}
```


### no_split_stack

{clang-attr-syntaxes}`NoSplitStackDocs`

The `no_split_stack` attribute disables the emission of the split stack
preamble for a particular function. It has no effect if `-fsplit-stack`
is not specified.


### no_stack_protector, safebuffers

{clang-attr-syntaxes}`NoStackProtectorDocs`

Clang supports the GNU style `__attribute__((no_stack_protector))` and Microsoft
style `__declspec(safebuffers)` attribute which disables
the stack protector on the specified function. This attribute is useful for
selectively disabling the stack protector on some functions when building with
`-fstack-protector` compiler option.

For example, it disables the stack protector for the function `foo` but function
`bar` will still be built with the stack protector with the `-fstack-protector`
option.

```c
int __attribute__((no_stack_protector))
foo (int x); // stack protection will be disabled for foo.

int bar(int y); // bar can be built with the stack protector.
```


### noalias

{clang-attr-syntaxes}`NoAliasDocs`

The `noalias` attribute indicates that the only memory accesses inside
function are loads and stores from objects pointed to by its pointer-typed
arguments, with arbitrary offsets.


### nocf_check

{clang-attr-syntaxes}`AnyX86NoCfCheckDocs`

Jump Oriented Programming attacks rely on tampering with addresses used by
indirect call / jmp, e.g. redirect control-flow to non-programmer
intended bytes in the binary.
X86 Supports Indirect Branch Tracking (IBT) as part of Control-Flow
Enforcement Technology (CET). IBT instruments ENDBR instructions used to
specify valid targets of indirect call / jmp.
The `nocf_check` attribute has two roles:
1\. Appertains to a function - do not add ENDBR instruction at the beginning of
the function.
2\. Appertains to a function pointer - do not track the target function of this
pointer (by adding nocf_check prefix to the indirect-call instruction).


### noconvergent

{clang-attr-syntaxes}`NoConvergentDocs`

This attribute prevents a function from being treated as convergent; when a
function is marked `noconvergent`, calls to that function are not
automatically assumed to be convergent, unless such calls are explicitly marked
as `convergent`. If a statement is marked as `noconvergent`, any calls to
inline `asm` in that statement are no longer treated as convergent.

In languages following SPMD/SIMT programming model, e.g., CUDA/HIP, function
declarations and inline asm calls are treated as convergent by default for
correctness. This `noconvergent` attribute is helpful for developers to
prevent them from being treated as convergent when it's safe.

```c
__device__ float bar(float);
__device__ float foo(float) __attribute__((noconvergent)) {}

__device__ int example(void) {
  float x;
  [[clang::noconvergent]] x = bar(x); // no effect on convergence
  [[clang::noconvergent]] { asm volatile ("nop"); } // the asm call is non-convergent
}
```


### nodiscard, warn_unused_result

{clang-attr-syntaxes}`WarnUnusedResultsDocs`

Clang supports the ability to diagnose when the results of a function call
expression are discarded under suspicious circumstances. A diagnostic is
generated when a function or its return type is marked with `[[nodiscard]]`
(or `__attribute__((warn_unused_result))`) and the function call appears as a
potentially-evaluated discarded-value expression that is not explicitly cast to
`void`.

A string literal may optionally be provided to the attribute, which will be
reproduced in any resulting diagnostics. Redeclarations using different forms
of the attribute (with or without the string literal or with different string
literal contents) are allowed. If there are redeclarations of the entity with
differing string literals, it is unspecified which one will be used by Clang
in any resulting diagnostics.

```c++
struct [[nodiscard]] error_info { /*...*/ };
error_info enable_missile_safety_mode();

void launch_missiles();
void test_missiles() {
  enable_missile_safety_mode(); // diagnoses
  launch_missiles();
}
error_info &foo();
void f() { foo(); } // Does not diagnose, error_info is a reference.
```

Additionally, discarded temporaries resulting from a call to a constructor
marked with `[[nodiscard]]` or a constructor of a type marked
`[[nodiscard]]` will also diagnose. This also applies to type conversions that
use the annotated `[[nodiscard]]` constructor or result in an annotated type.

```c++
struct [[nodiscard]] marked_type {/*..*/ };
struct marked_ctor {
  [[nodiscard]] marked_ctor();
  marked_ctor(int);
};

struct S {
  operator marked_type() const;
  [[nodiscard]] operator int() const;
};

void usages() {
  marked_type(); // diagnoses.
  marked_ctor(); // diagnoses.
  marked_ctor(3); // Does not diagnose, int constructor isn't marked nodiscard.

  S s;
  static_cast<marked_type>(s); // diagnoses
  (int)s; // diagnoses
}
```


### noduplicate

{clang-attr-syntaxes}`NoDuplicateDocs`

The `noduplicate` attribute can be placed on function declarations to control
whether function calls to this function can be duplicated or not as a result of
optimizations. This is required for the implementation of functions with
certain special requirements, like the OpenCL "barrier" function, that might
need to be run concurrently by all the threads that are executing in lockstep
on the hardware. For example this attribute applied on the function
`nodupfunc` in the code below avoids that:

```c
void nodupfunc() __attribute__((noduplicate));
// Setting it as a C++11 attribute is also valid
// void nodupfunc() [[clang::noduplicate]];
void foo();
void bar();

nodupfunc();
if (a > n) {
  foo();
} else {
  bar();
}
```

gets possibly modified by some optimizations into code similar to this:

```c
if (a > n) {
  nodupfunc();
  foo();
} else {
  nodupfunc();
  bar();
}
```

where the call to `nodupfunc` is duplicated and sunk into the two branches
of the condition.


### noinline

{clang-attr-syntaxes}`NoInlineDocs`

This function attribute suppresses the inlining of a function at the call sites
of the function.

`[[clang::noinline]]` spelling can be used as a statement attribute; other
spellings of the attribute are not supported on statements. If a statement is
marked `[[clang::noinline]]` and contains calls, those calls inside the
statement will not be inlined by the compiler.

`__noinline__` can be used as a keyword in CUDA/HIP languages. This is to
avoid diagnostics due to usage of `__attribute__((__noinline__))`
with `__noinline__` defined as a macro as `__attribute__((noinline))`.

```c
int example(void) {
  int r;
  [[clang::noinline]] foo();
  [[clang::noinline]] r = bar();
  return r;
}
```


### noreturn, _Noreturn

{clang-attr-syntaxes}`CXX11NoReturnDocs`

A function declared as `[[noreturn]]` shall not return to its caller. The
compiler will generate a diagnostic for a function declared as `[[noreturn]]`
that appears to be capable of returning to its caller.

The `[[_Noreturn]]` spelling is deprecated and only exists to ease code
migration for code using `[[noreturn]]` after including `<stdnoreturn.h>`.


### not_tail_called

{clang-attr-syntaxes}`NotTailCalledDocs`

The `not_tail_called` attribute prevents tail-call optimization on statically
bound calls. Objective-c methods, and functions marked as `always_inline`
cannot be marked as `not_tail_called`.

For example, it prevents tail-call optimization in the following case:

```c
int __attribute__((not_tail_called)) foo1(int);

int foo2(int a) {
  return foo1(a); // No tail-call optimization on direct calls.
}
```

However, it doesn't prevent tail-call optimization in this case:

```c
int __attribute__((not_tail_called)) foo1(int);

int foo2(int a) {
  int (*fn)(int) = &foo1;

  // not_tail_called has no effect on an indirect call even if the call can
  // be resolved at compile time.
  return (*fn)(a);
}
```

Generally, marking an overriding virtual function as `not_tail_called` is
not useful, because this attribute is a property of the static type. Calls
made through a pointer or reference to the base class type will respect
the `not_tail_called` attribute of the base class's member function,
regardless of the runtime destination of the call:

```c++
struct Foo { virtual void f(); };
struct Bar : Foo {
  [[clang::not_tail_called]] void f() override;
};
void callera(Bar& bar) {
  Foo& foo = bar;
  // not_tail_called has no effect on here, even though the
  // underlying method is f from Bar.
  foo.f();
  bar.f(); // No tail-call optimization on here.
}
```


### nothrow

{clang-attr-syntaxes}`NoThrowDocs`

Clang supports the GNU style `__attribute__((nothrow))` and Microsoft style
`__declspec(nothrow)` attribute as an equivalent of `noexcept` on function
declarations. This attribute informs the compiler that the annotated function
does not throw an exception. This prevents exception-unwinding. This attribute
is particularly useful on functions in the C Standard Library that are
guaranteed to not throw an exception.


### nouwtable

{clang-attr-syntaxes}`NoUwtableDocs`

Clang supports the `nouwtable` attribute which skips emitting
the unwind table entry for the specified function. This attribute is useful for
selectively emitting the unwind table entry on some functions when building with
`-funwind-tables` compiler option.


### numthreads

{clang-attr-syntaxes}`NumThreadsDocs`

The `numthreads` attribute applies to HLSL shaders where explcit thread counts
are required. The `X`, `Y`, and `Z` values provided to the attribute
dictate the thread id. Total number of threads executed is `X * Y * Z`.

The full documentation is available here: <https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-attributes-numthreads>


### objc_method_family

{clang-attr-syntaxes}`ObjCMethodFamilyDocs`

Many methods in Objective-C have conventional meanings determined by their
selectors. It is sometimes useful to be able to mark a method as having a
particular conventional meaning despite not having the right selector, or as
not having the conventional meaning that its selector would suggest. For these
use cases, we provide an attribute to specifically describe the "method family"
that a method belongs to.

**Usage**: `__attribute__((objc_method_family(X)))`, where `X` is one of
`none`, `alloc`, `copy`, `init`, `mutableCopy`, or `new`. This
attribute can only be placed at the end of a method declaration:

```objc
- (NSString *)initMyStringValue __attribute__((objc_method_family(none)));
```

Users who do not wish to change the conventional meaning of a method, and who
merely want to document its non-standard retain and release semantics, should
use the retaining behavior attributes (`ns_returns_retained`,
`ns_returns_not_retained`, etc).

Query for this feature with `__has_attribute(objc_method_family)`.


### objc_requires_super

{clang-attr-syntaxes}`ObjCRequiresSuperDocs`

Some Objective-C classes allow a subclass to override a particular method in a
parent class but expect that the overriding method also calls the overridden
method in the parent class. For these cases, we provide an attribute to
designate that a method requires a "call to `super`" in the overriding
method in the subclass.

**Usage**: `__attribute__((objc_requires_super))`. This attribute can only
be placed at the end of a method declaration:

```objc
- (void)foo __attribute__((objc_requires_super));
```

This attribute can only be applied the method declarations within a class, and
not a protocol. Currently this attribute does not enforce any placement of
where the call occurs in the overriding method (such as in the case of
`-dealloc` where the call must appear at the end). It checks only that it
exists.

Note that on both OS X and iOS that the Foundation framework provides a
convenience macro `NS_REQUIRES_SUPER` that provides syntactic sugar for this
attribute:

```objc
- (void)foo NS_REQUIRES_SUPER;
```

This macro is conditionally defined depending on the compiler's support for
this attribute. If the compiler does not support the attribute the macro
expands to nothing.

Operationally, when a method has this annotation the compiler will warn if the
implementation of an override in a subclass does not call super. For example:

```objc
warning: method possibly missing a [super AnnotMeth] call
- (void) AnnotMeth{};
                   ^
```


### optnone

{clang-attr-syntaxes}`OptnoneDocs`

The `optnone` attribute suppresses essentially all optimizations
on a function or method, regardless of the optimization level applied to
the compilation unit as a whole. This is particularly useful when you
need to debug a particular function, but it is infeasible to build the
entire application without optimization. Avoiding optimization on the
specified function can improve the quality of the debugging information
for that function.

This attribute is incompatible with the `always_inline` and `minsize`
attributes.

Note that this attribute does not apply recursively to nested functions such as
lambdas or blocks when using declaration-specific attribute syntaxes such as double
square brackets (`[[]]`) or `__attribute__`. The `#pragma` syntax can be
used to apply the attribute to all functions, including nested functions, in a
range of source code.


### overloadable

{clang-attr-syntaxes}`OverloadableDocs`

Clang provides support for C++ function overloading in C. Function overloading
in C is introduced using the `overloadable` attribute. For example, one
might provide several overloaded versions of a `tgsin` function that invokes
the appropriate standard function computing the sine of a value with `float`,
`double`, or `long double` precision:

```c
#include <math.h>
float __attribute__((overloadable)) tgsin(float x) { return sinf(x); }
double __attribute__((overloadable)) tgsin(double x) { return sin(x); }
long double __attribute__((overloadable)) tgsin(long double x) { return sinl(x); }
```

Given these declarations, one can call `tgsin` with a `float` value to
receive a `float` result, with a `double` to receive a `double` result,
etc. Function overloading in C follows the rules of C++ function overloading
to pick the best overload given the call arguments, with a few C-specific
semantics:

- Conversion from `float` or `double` to `long double` is ranked as a
  floating-point promotion (per C99) rather than as a floating-point conversion
  (as in C++).
- A conversion from a pointer of type `T*` to a pointer of type `U*` is
  considered a pointer conversion (with conversion rank) if `T` and `U` are
  compatible types.
- A conversion from type `T` to a value of type `U` is permitted if `T`
  and `U` are compatible types. This conversion is given "conversion" rank.
- If no viable candidates are otherwise available, we allow a conversion from a
  pointer of type `T*` to a pointer of type `U*`, where `T` and `U` are
  incompatible. This conversion is ranked below all other types of conversions.
  Please note: `U` lacking qualifiers that are present on `T` is sufficient
  for `T` and `U` to be incompatible.

The declaration of `overloadable` functions is restricted to function
declarations and definitions. If a function is marked with the `overloadable`
attribute, then all declarations and definitions of functions with that name,
except for at most one (see the note below about unmarked overloads), must have
the `overloadable` attribute. In addition, redeclarations of a function with
the `overloadable` attribute must have the `overloadable` attribute, and
redeclarations of a function without the `overloadable` attribute must *not*
have the `overloadable` attribute. e.g.,

```c
int f(int) __attribute__((overloadable));
float f(float); // error: declaration of "f" must have the "overloadable" attribute
int f(int); // error: redeclaration of "f" must have the "overloadable" attribute

int g(int) __attribute__((overloadable));
int g(int) { } // error: redeclaration of "g" must also have the "overloadable" attribute

int h(int);
int h(int) __attribute__((overloadable)); // error: declaration of "h" must not
                                          // have the "overloadable" attribute
```

Functions marked `overloadable` must have prototypes. Therefore, the
following code is ill-formed:

```c
int h() __attribute__((overloadable)); // error: h does not have a prototype
```

However, `overloadable` functions are allowed to use a ellipsis even if there
are no named parameters (as is permitted in C++). This feature is particularly
useful when combined with the `unavailable` attribute:

```c++
void honeypot(...) __attribute__((overloadable, unavailable)); // calling me is an error
```

Functions declared with the `overloadable` attribute have their names mangled
according to the same rules as C++ function names. For example, the three
`tgsin` functions in our motivating example get the mangled names
`_Z5tgsinf`, `_Z5tgsind`, and `_Z5tgsine`, respectively. There are two
caveats to this use of name mangling:

- Future versions of Clang may change the name mangling of functions overloaded
  in C, so you should not depend on an specific mangling. To be completely
  safe, we strongly urge the use of `static inline` with `overloadable`
  functions.
- The `overloadable` attribute has almost no meaning when used in C++,
  because names will already be mangled and functions are already overloadable.
  However, when an `overloadable` function occurs within an `extern "C"`
  linkage specification, its name *will* be mangled in the same way as it
  would in C.

For the purpose of backwards compatibility, at most one function with the same
name as other `overloadable` functions may omit the `overloadable`
attribute. In this case, the function without the `overloadable` attribute
will not have its name mangled.

For example:

```c
// Notes with mangled names assume Itanium mangling.
int f(int);
int f(double) __attribute__((overloadable));
void foo() {
  f(5); // Emits a call to f (not _Z1fi, as it would with an overload that
        // was marked with overloadable).
  f(1.0); // Emits a call to _Z1fd.
}
```

Support for unmarked overloads is not present in some versions of clang. You may
query for it using `__has_extension(overloadable_unmarked)`.

Query for this attribute with `__has_attribute(overloadable)`.


(analyzer-ownership-attrs)=

### ownership_holds, ownership_returns, ownership_takes (Clang Static Analyzer)

{clang-attr-syntaxes}`OwnershipDocs`

:::{note}
In order for the Clang Static Analyzer to acknowledge these attributes, the
`Optimistic` config needs to be set to true for the checker
`unix.DynamicMemoryModeling`:

`-Xclang -analyzer-config -Xclang unix.DynamicMemoryModeling:Optimistic=true`
:::

These attributes are used by the Clang Static Analyzer's dynamic memory modeling
facilities to mark custom allocating/deallocating functions.

All 3 attributes' first parameter of type string is the type of the allocation:
`malloc`, `new`, etc. to allow for catching {ref}`mismatched deallocation
<unix-MismatchedDeallocator>` bugs. The allocation type can be any string, e.g.
a function annotated with
returning a piece of memory of type `lasagna` but freed with a function
annotated to release `cheese` typed memory will result in mismatched
deallocation warning.

The (currently) only allocation type having special meaning is `malloc` --
the Clang Static Analyzer makes sure that allocating functions annotated with
`malloc` are treated like they used the standard `malloc()`, and can be
safely deallocated with the standard `free()`.

- Use `ownership_returns` to mark a function as an allocating function.
  It takes 1 or 2 arguments.
  The first argument is a user-provided identifier representing the "kind" of the allocation.
  This is basically what is enforced when checking the deallocation. This is mandatory.
  The second argument is optional.
  It represents the index of the parameter that represents the allocation size in bytes (counting from 1).
  The referenced parameter must have some integral type.
  This attribute may appear at most once per declaration.
  If this argument is not set, then tooling, such as the Clang Static Analyzer,
  won't be able to reason about the size of the allocation, thus check potential out-of-bounds accesses.
  However, such tooling could still warn if the wrong deallocation function
  was used for the `ownership_returns` attributed resource.
  If forward declarations have this attribute, those must have the same arguments.
- Use `ownership_takes` to mark a function as a deallocating function. Takes 2
  arguments: the allocation type, and the index of the parameter that is being
  deallocated (counting from 1).
- Use `ownership_holds` to mark that a function takes over the ownership of a
  piece of memory and will free it at some unspecified point in the future. Like
  `ownership_takes`, this takes 2 arguments: the allocation type, and the
  index of the parameter whose ownership will be taken over (counting from 1).

The annotations `ownership_takes` and `ownership_holds` both prevent memory
leak reports (concerning the specified parameter); the difference between them
is that using taken memory is a use-after-free error, while using held memory
is assumed to be legitimate. However, releasing the held memory or passing it
to another holding call is reported by the analyzer as an "attempt to release
non-owned memory".

Example:

```c
// Denotes that my_malloc will return with a dynamically allocated piece of
// memory using malloc().
void __attribute((ownership_returns(malloc))) *my_malloc(size_t sz);

// 'sz' (parameter 1) is the allocation size.
void __attribute((ownership_returns(malloc, 1))) *my_sized_malloc(size_t sz);

// Denotes that my_free will deallocate its argument using free().
void __attribute((ownership_takes(malloc, 1))) my_free(void *);

// Denotes that my_hold will take over the ownership of its argument that was
// allocated via malloc().
void __attribute((ownership_holds(malloc, 1))) my_hold(void *);
```

Further reading about dynamic memory modeling in the Clang Static Analyzer is
found in these checker docs:
{ref}`unix.Malloc <unix-Malloc>`, {ref}`unix.MallocSizeof <unix-MallocSizeof>`,
{ref}`unix.MismatchedDeallocator <unix-MismatchedDeallocator>`,
{ref}`cplusplus.NewDelete <cplusplus-NewDelete>`,
{ref}`cplusplus.NewDeleteLeaks <cplusplus-NewDeleteLeaks>`,
{ref}`optin.taint.TaintedAlloc <optin-taint-TaintedAlloc>`.
Mind that many more checkers are affected by dynamic memory modeling changes to
some extent.

Further reading for other annotations:
{doc}`Static Analyzer source annotations <analyzer/user-docs/Annotations>`.


### packoffset

{clang-attr-syntaxes}`HLSLPackOffsetDocs`

The packoffset attribute is used to change the layout of a cbuffer.
Attribute spelling in HLSL is: `packoffset( c[Subcomponent][.component] )`.
A subcomponent is a register number, which is an integer. A component is in the form of [.xyzw].

Examples:

```hlsl
cbuffer A {
  float3 a : packoffset(c0.y);
  float4 b : packoffset(c4);
}
```

The full documentation is available here: <https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-variable-packoffset>


### patchable_function_entry

{clang-attr-syntaxes}`PatchableFunctionEntryDocs`

`__attribute__((patchable_function_entry(N,M,Section)))` is used to generate M
NOPs before the function entry and N-M NOPs after the function entry, with a record of
the entry stored in section `Section`. This attribute takes precedence over the
command line option `-fpatchable-function-entry=N,M,Section`. `M` defaults to 0
if omitted. `Section` defaults to the `-fpatchable-function-entry` section name if
set, or to `__patchable_function_entries` otherwise.

This attribute is only supported on
aarch64/aarch64-be/loongarch32/loongarch64/riscv32/riscv64/i386/x86-64/ppc/ppc64/ppc64le/s390x targets.
For ppc/ppc64 targets, AIX is still not supported.


### personality

{clang-attr-syntaxes}`PersonalityDocs`

`__attribute__((personality(<routine>)))` is used to specify a personality
routine that is different from the language that is being used to implement the
function. This is a targeted, low-level feature aimed at language runtime
implementors who write runtime support code in C/C++ but need that code to
participate in a foreign language's exception-handling or unwinding model.

A personality routine is a language-specific callback attached to each stack
frame that the unwinder invokes to determine whether that frame handles a given
exception and what cleanup actions to perform. It effectively colors the
language-agnostic unwinding mechanism with language-specific semantics, enabling
different languages to coexist on the same call stack while each interpreting
exceptions according to their own rules.


### preserve_access_index

{clang-attr-syntaxes}`BPFPreserveAccessIndexDocs`

Clang supports the `__attribute__((preserve_access_index))`
attribute for the BPF target. This attribute may be attached to a
struct or union declaration, where if -g is specified, it enables
preserving struct or union member access debuginfo indices of this
struct or union, similar to clang `__builtin_preserve_access_index()`.


### preserve_static_offset

{clang-attr-syntaxes}`BPFPreserveStaticOffsetDocs`

Clang supports the `__attribute__((preserve_static_offset))`
attribute for the BPF target. This attribute may be attached to a
struct or union declaration. Reading or writing fields of types having
such annotation is guaranteed to generate LDX/ST/STX instruction with
offset corresponding to the field.

For example:

```c
struct foo {
  int a;
  int b;
};

struct bar {
  int a;
  struct foo b;
} __attribute__((preserve_static_offset));

void buz(struct bar *g) {
  g->b.a = 42;
}
```

The assignment to `g`'s field would produce an ST instruction with
offset 8: `*(u32)(r1 + 8) = 42;`.

Without this attribute generated instructions might be different,
depending on optimizations behavior. E.g. the example above could be
rewritten as `r1 += 8; *(u32)(r1 + 0) = 42;`.


### register

{clang-attr-syntaxes}`HLSLResourceBindingDocs`

The resource binding attribute sets the virtual register and logical register space for a resource.
Attribute spelling in HLSL is: `register(slot [, space])`.
`slot` takes the format `[type][number]`,
where `type` is a single character specifying the resource type and `number` is the virtual register number.

Register types are:
t for shader resource views (SRV),
s for samplers,
u for unordered access views (UAV),
b for constant buffer views (CBV).

Register space is specified in the format `space[number]` and defaults to `space0` if omitted.
Here're resource binding examples with and without space:

```hlsl
RWBuffer<float> Uav : register(u3, space1);
Buffer<float> Buf : register(t1);
```

The full documentation is available here: <https://docs.microsoft.com/en-us/windows/win32/direct3d12/resource-binding-in-hlsl>


### reinitializes

{clang-attr-syntaxes}`ReinitializesDocs`

The `reinitializes` attribute can be applied to a non-static, non-const C++
member function to indicate that this member function reinitializes the entire
object to a known state, independent of the previous state of the object.

This attribute can be interpreted by static analyzers that warn about uses of an
object that has been left in an indeterminate state by a move operation. If a
member function marked with the `reinitializes` attribute is called on a
moved-from object, the analyzer can conclude that the object is no longer in an
indeterminate state.

A typical example where this attribute would be used is on functions that clear
a container class:

```c++
template <class T>
class Container {
public:
  ...
  [[clang::reinitializes]] void Clear();
  ...
};
```


### release_capability, release_shared_capability

{clang-attr-syntaxes}`ReleaseCapabilityDocs`

Marks a function as releasing a capability.


### retain

{clang-attr-syntaxes}`RetainDocs`

This attribute, when attached to a function or variable definition, prevents
section garbage collection in the linker. It does not prevent other discard
mechanisms, such as archive member selection, and COMDAT group resolution.

If the compiler does not emit the definition, e.g. because it was not used in
the translation unit or the compiler was able to eliminate all of the uses,
this attribute has no effect. This attribute is typically combined with the
`used` attribute to force the definition to be emitted and preserved into the
final linked image.

This attribute is only necessary on ELF targets; other targets prevent section
garbage collection by the linker when using the `used` attribute alone.
Using the attributes together should result in consistent behavior across
targets.

This attribute requires the linker to support the `SHF_GNU_RETAIN` extension.
This support is available in GNU `ld` and `gold` as of binutils 2.36, as
well as in `ld.lld` 13.


### sentinel

{clang-attr-syntaxes}`SentinelDocs`

The `sentinel` attribute can be applied to variadic functions and pointers to
variadic functions, to diagnose each function call that does not pass a
sentinel value (a null pointer constant) as the last argument to the function
call. The attribute accepts two optional arguments: the first argument is the
position of the expected sentinel value, starting from the last parameter. The
second argument describes whether the last fixed parameter is treated as a
valid sentinel value when set to `1`.
All arguments described above default to `0` when elided.
The attribute is also supported with blocks and in Objective-C.

```c
void foo(const char*, ...) __attribute__((sentinel));
void bar(int, ...) __attribute__((sentinel(1)));
void baz(const char*, const char*, ...) __attribute__((sentinel(0, 1)));

void example() {
  foo("Example", (void*)0);
  foo("Another", "example", NULL);
  foo("Missing", "sentinel"); // Not OK

  bar(1, 2, NULL, 3);         // OK: sentinel value at the 2nd to last position
  bar(1, 2, 3, nullptr, 4);   // OK: `nullptr` is valid in C23
  bar(1, 2, 3, 4, NULL);      // Not OK

  baz("Test", "with", "multiple", "args", NULL);
  baz("One", NULL);           // OK: last fixed parameter is a valid sentinel

  void (*ptr) (int arg, ...) __attribute__ ((__sentinel__));
  ptr(1, 2, 3, NULL);
}
```

```c++
struct Ty {
    int value;

    template<typename T>
    auto&& foo(T&& val, ...) __attribute__((sentinel(1))) {
        return std::forward<T>(val);
    }

    template<class Self>
    auto&& bar(this Self&& self, ...) __attribute__((sentinel(1))) {
        return std::forward<Self>(self).value;
    }
};

void example2() {
    auto sty = Ty{};
    sty.foo(1, nullptr, 3);
    sty.bar(1, nullptr, 3);

    auto lmbd = [](int a, ...) __attribute__((sentinel)) {};
    lmbd(1, 2, nullptr);
}
```


### shader

{clang-attr-syntaxes}`HLSLSV_ShaderTypeAttrDocs`

The `shader` type attribute applies to HLSL shader entry functions to
identify the shader type for the entry function.
The syntax is:

```text
[shader(string-literal)]
```

where the string literal is one of: "pixel", "vertex", "geometry", "hull",
"domain", "compute", "raygeneration", "intersection", "anyhit", "closesthit",
"miss", "callable", "mesh", "amplification". Normally the shader type is set
by shader target with the `-T` option like `-Tps_6_1`. When compiling to a
library target like `lib_6_3`, the shader type attribute can help the
compiler to identify the shader type. It is mostly used by Raytracing shaders
where shaders must be compiled into a library and linked at runtime.


### short_call, near

{clang-attr-syntaxes}`MipsShortCallStyleDocs`

Clang supports the `__attribute__((long_call))`, `__attribute__((far))`,
`__attribute__((short__call))`, and `__attribute__((near))` attributes
on MIPS targets. These attributes may only be added to function declarations
and change the code generated by the compiler when directly calling
the function. The `short_call` and `near` attributes are synonyms and
allow calls to the function to be made using the `jal` instruction, which
requires the function to be located in the same naturally aligned 256MB segment
as the caller. The `long_call` and `far` attributes are synonyms and
require the use of a different call sequence that works regardless
of the distance between the functions.

These attributes have no effect for position-independent code.

These attributes take priority over command line switches such
as `-mlong-calls` and `-mno-long-calls`.


### signal

{clang-attr-syntaxes}`AVRSignalDocs`

Clang supports the GNU style `__attribute__((signal))` attribute on
AVR targets. This attribute may be attached to a function definition and instructs
the backend to generate appropriate function entry/exit code so that it can be used
directly as an interrupt service routine.

Interrupt handler functions defined with the signal attribute do not re-enable interrupts.


### speculative_load_hardening

{clang-attr-syntaxes}`SpeculativeLoadHardeningDocs`

This attribute can be applied to a function declaration in order to indicate
that [Speculative Load Hardening][slh]
should be enabled for the function body. This can also be applied to a method
in Objective C. This attribute will take precedence over the command line flag
in the case where {option}`-mno-speculative-load-hardening` is specified.

[slh]: https://llvm.org/docs/SpeculativeLoadHardening.html

Speculative Load Hardening is a best-effort mitigation against
information leak attacks that make use of control flow
miss-speculation - specifically miss-speculation of whether a branch
is taken or not. Typically vulnerabilities enabling such attacks are
classified as "Spectre variant #1". Notably, this does not attempt to
mitigate against miss-speculation of branch target, classified as
"Spectre variant #2" vulnerabilities.

When inlining, the attribute is sticky. Inlining a function that
carries this attribute will cause the caller to gain the
attribute. This is intended to provide a maximally conservative model
where the code in a function annotated with this attribute will always
(even after inlining) end up hardened.


### stack_protector_ignore

{clang-attr-syntaxes}`StackProtectorIgnoreDocs`

The `stack_protector_ignore` attribute skips analysis of the given local
variable when determining if a function should use a stack protector.

The `-fstack-protector` option uses a heuristic to only add stack protectors
to functions which contain variables or buffers over some size threshold. This
attribute overrides that heuristic for the attached variable, opting
them out. If this results in no variables or buffers remaining over the stack
protector threshold, then the function will no longer use a stack protector.


### strict_gs_check

{clang-attr-syntaxes}`StrictGuardStackCheckDocs`

Clang supports the Microsoft style `__declspec((strict_gs_check))` attribute
which upgrades the stack protector check from `-fstack-protector` to
`-fstack-protector-strong`.

For example, it upgrades the stack protector for the function `foo` to
`-fstack-protector-strong` but function `bar` will still be built with the
stack protector with the `-fstack-protector` option.

```c
__declspec((strict_gs_check))
int foo(int x); // stack protection will be upgraded for foo.

int bar(int y); // bar can be built with the standard stack protector checks.
```


### sycl_external

{clang-attr-syntaxes}`SYCLExternalDocs`

The `sycl_external` attribute indicates that a function defined in another
translation unit may be called by a device function defined in the current
translation unit or, if defined in the current translation unit, the function
may be called by device functions defined in other translation units.
The attribute is intended for use in the implementation of the `SYCL_EXTERNAL`
macro as specified in section 5.10.1, "SYCL functions and member functions
linkage", of the SYCL 2020 specification.

The attribute only appertains to functions and only those that meet the
following requirements:

- Has external linkage
- Is not explicitly defined as deleted (the function may be an explicitly
  defaulted function that is defined as deleted)

The attribute shall be present on the first declaration of a function and
may optionally be present on subsequent declarations.

When compiling for a SYCL device target that does not support the generic
address space, the function shall not specify a raw pointer or reference type
as the return type or as a parameter type.
See section 5.10, "SYCL offline linking", of the SYCL 2020 specification.
The following examples demonstrate the use of this attribute:

```c++
[[clang::sycl_external]] void Foo(); // Ok.

[[clang::sycl_external]] void Bar() { /* ... */ } // Ok.

[[clang::sycl_external]] extern void Baz(); // Ok.

[[clang::sycl_external]] static void Quux() { /* ... */ } // error:  Quux() has internal linkage.
```


### sycl_kernel

{clang-attr-syntaxes}`SYCLKernelDocs`

The `sycl_kernel` attribute specifies that a function template will be used
to outline device code and to generate an OpenCL kernel.
Here is a code example of the SYCL program, which demonstrates the compiler's
outlining job:

```c++
int foo(int x) { return ++x; }

using namespace cl::sycl;
queue Q;
buffer<int, 1> a(range<1>{1024});
Q.submit([&](handler& cgh) {
  auto A = a.get_access<access::mode::write>(cgh);
  cgh.parallel_for<init_a>(range<1>{1024}, [=](id<1> index) {
    A[index] = index[0] + foo(42);
  });
}
```

A C++ function object passed to the `parallel_for` is called a "SYCL kernel".
A SYCL kernel defines the entry point to the "device part" of the code. The
compiler will emit all symbols accessible from a "kernel". In this code
example, the compiler will emit "foo" function. More details about the
compilation of functions for the device part can be found in the SYCL 1.2.1
specification Section 6.4.
To show to the compiler entry point to the "device part" of the code, the SYCL
runtime can use the `sycl_kernel` attribute in the following way:

```c++
namespace cl {
namespace sycl {
class handler {
  template <typename KernelName, typename KernelType/*, ...*/>
  __attribute__((sycl_kernel)) void sycl_kernel_function(KernelType KernelFuncObj) {
    // ...
    KernelFuncObj();
  }

  template <typename KernelName, typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    sycl_kernel_function<KernelName, KernelType, Dims>(KernelFunc);
#else
    // Host implementation
#endif
  }
};
} // namespace sycl
} // namespace cl
```

The compiler will also generate an OpenCL kernel using the function marked with
the `sycl_kernel` attribute.
Here is the list of SYCL device compiler expectations with regard to the
function marked with the `sycl_kernel` attribute:

- The function must be a template with at least two type template parameters.
  The compiler generates an OpenCL kernel and uses the first template parameter
  as a unique name for the generated OpenCL kernel. The host application uses
  this unique name to invoke the OpenCL kernel generated for the SYCL kernel
  specialized by this name and second template parameter `KernelType` (which
  might be an unnamed function object type).
- The function must have at least one parameter. The first parameter is
  required to be a function object type (named or unnamed i.e. lambda). The
  compiler uses function object type fields to generate OpenCL kernel
  parameters.
- The function must return void. The compiler reuses the body of marked functions to
  generate the OpenCL kernel body, and the OpenCL kernel must return `void`.

The SYCL kernel in the previous code sample meets these expectations.


### sycl_kernel_entry_point

{clang-attr-syntaxes}`SYCLKernelEntryPointDocs`

The `sycl_kernel_entry_point` attribute facilitates the launch of a SYCL
kernel and the generation of an offload kernel entry point, sometimes called
a SYCL kernel caller function, suitable for invoking a SYCL kernel on an
offload device. The attribute is intended for use in the implementation of
SYCL kernel invocation functions like the `single_task` and `parallel_for`
member functions of the `sycl::handler` class specified in section 4.9.4,
"Command group `handler` class", of the SYCL 2020 specification.

The attribute requires a single type argument that meets the requirements for
a SYCL kernel name as described in section 5.2, "Naming of kernels", of the
SYCL 2020 specification. A unique kernel name type is required for each
function declared with the attribute. The attribute may not first appear on a
declaration that follows a definition of the function.

The attribute only appertains to functions and only those that meet the
following requirements.

- Has a non-deduced `void` return type.
- Is not a constructor or destructor.
- Is not a non-static member function with an explicit object parameter.
- Is not a C variadic function.
- Is not a coroutine.
- Is not defined as deleted or as defaulted.
- Is not defined with a function try block.
- Is not declared with the `constexpr` or `consteval` specifiers.
- Is not declared with the `[[noreturn]]` attribute.

Use in the implementation of a SYCL kernel invocation function might look as
follows.

```c++
namespace sycl {
class handler {
  template<typename KernelName, typename... Ts>
  void sycl_kernel_launch(const char* kernelSymbol, Ts&&... kernelArgs) {
    // This code will run on the host and is responsible for calling functions
    // appropriate for the desired offload backend (OpenCL, CUDA, HIP,
    // Level Zero, etc...) to copy the kernel arguments denoted by kernelArgs
    // to a device and to schedule an invocation of the offload kernel entry
    // point denoted by kernelSymbol with the copied arguments.
  }

  template<typename KernelName, typename KernelType>
  [[ clang::sycl_kernel_entry_point(KernelName) ]]
  void kernel_entry_point(KernelType kernelFunc) {
    // This code will run on the device. The call to kernelFunc() invokes
    // the SYCL kernel.
    kernelFunc();
  }

public:
  template<typename KernelName, typename KernelType>
  void single_task(const KernelType& kernelFunc) {
    // This code will run on the host. kernel_entry_point() is called to
    // trigger generation of an offload kernel entry point and to schedule
    // an invocation of it on a device with kernelFunc (a SYCL kernel object)
    // passed as a kernel argument. This call will result in an implicit call
    // to sycl_kernel_launch() with the symbol name for the generated offload
    // kernel entry point passed as the first function argument followed by
    // kernelFunc.
    kernel_entry_point<KernelName>(kernelFunc);
  }
};
} // namespace sycl
```

A SYCL kernel object is a callable object of class type that is constructed on
a host, often via a lambda expression, and then passed to a SYCL kernel
invocation function to be executed on an offload device. The `kernelFunc`
parameters in the example code above correspond to SYCL kernel objects.

A SYCL kernel object type is required to satisfy the device copyability
requirements specified in section 3.13.1, "Device copyable", of the SYCL 2020
specification. Additionally, any data members of the kernel object type are
required to satisfy section 4.12.4, "Rules for parameter passing to kernels".
For most types, these rules require that the type is trivially copyable.
However, the SYCL specification mandates that certain special SYCL types, such
as `sycl::accessor` and `sycl::stream`, be device copyable even if they are
not trivially copyable. These types require special handling because they cannot
necessarily be copied to device memory as if by `memcpy()`.

The SYCL kernel object and its data members constitute the parameters of an
offload kernel. An offload kernel consists of an offload entry point function
and the set of all functions and variables that are directly or indirectly used
by the entry point function.

A SYCL kernel invocation function is responsible for performing the following
tasks (likely with the help of an offload backend like OpenCL):

1. Identifying the offload kernel entry point to be used for the SYCL kernel.
2. Validating that the SYCL kernel object type and its data members meet the
   SYCL device copyability and kernel parameter requirements noted above.
3. Copying the SYCL kernel object and any other kernel arguments to device
   memory including any special handling required for SYCL special types.
4. Initiating execution of the offload kernel entry point.

The offload kernel entry point for a SYCL kernel performs the following tasks:

1. Calling the `operator()` member function of the SYCL kernel object.

The `sycl_kernel_entry_point` attribute facilitates or automates these tasks
by providing generation of an offload kernel entry point with a unique symbol
name, type checking of kernel argument requirements, and initiation of kernel
execution via synthesized calls to a `sycl_kernel_launch` template.

A function declared with the `sycl_kernel_entry_point` attribute specifies
the parameters and body of an offload entry point function. Consider the
following call to the `single_task()` SYCL kernel invocation function assuming
an implementation similar to the one shown above.

```c++
struct S { int i; };
void f(sycl::handler &handler, sycl::stream &sout, S s) {
  handler.single_task<struct KN>([=] {
    sout << "The value of s.i is " << s.i << "\n";
  });
}
```

The SYCL kernel object is the result of the lambda expression. The call to
`kernel_entry_point()` via the call to `single_task()` triggers the
generation of an offload kernel entry point function that looks approximately
as follows.

```c++
void sycl-kernel-caller-for-KN(kernel-type kernelFunc) {
  kernelFunc();
}
```

There are a few items worthy of note:

1. `sycl-kernel-caller-for-KN` is an exposition only name; the actual name
   generated for an entry point is an implementation detail and subject to
   change. However, the name will incorporate the SYCL kernel name, `KN`,
   that was passed as the `KernelName` template parameter to
   `single_task()` and eventually provided as the argument to the
   `sycl_kernel_entry_point` attribute in order to ensure that a unique
   name is generated for each entry point. There is a one-to-one correspondence
   between SYCL kernel names and offload kernel entry points.
2. The SYCL kernel is a lambda closure type and therefore has no name;
   `kernel-type` is substituted above and corresponds to the `KernelType`
   template parameter deduced in the call to `single_task()`.
3. The parameter and the call to `kernelFunc()` in the function body
   correspond to the definition of `kernel_entry_point()` as called by
   `single_task()`.
4. The parameter is type checked for conformance with the SYCL device
   copyability and kernel parameter requirements.

Within `single_task()`, the call to `kernel_entry_point()` is effectively
replaced with a synthesized call to a `sycl_kernel_launch` template that
looks approximately as follows.

```c++
sycl_kernel_launch<KN>("sycl-kernel-caller-for-KN", kernelFunc);
```

There are a few items worthy of note:

1. Lookup for the `sycl_kernel_launch` template is performed as if from the
   body of the (possibly instantiated) definition of `kernel_entry_point()`.
   If name lookup or overload resolution fails, the program is ill-formed.
   If the selected overload is a non-static member function, then `this` is
   passed as the implicit object parameter.
2. Function arguments passed to `sycl_kernel_launch()` are passed
   as if by `std::move(x)`.
3. The `sycl_kernel_launch` template is expected to be provided by the SYCL
   library implementation. It is responsible for copying the kernel arguments
   to device memory and for scheduling execution of the generated offload
   kernel entry point identified by the symbol name passed as the first
   function argument. `sycl-kernel-caller-for-KN` is substituted above for
   the actual symbol name that would be generated for the offload kernel entry
   point.

It is not necessary for a function declared with the `sycl_kernel_entry_point`
attribute to be called for the offload kernel entry point to be emitted. For
inline functions and function templates, any ODR-use will suffice. For other
functions, an ODR-use is not required; the offload kernel entry point will be
emitted if the function is defined. In any case, a call to the function is
required for the synthesized call to `sycl_kernel_launch()` to occur.

A function declared with the `sycl_kernel_entry_point` attribute may include
an exception specification. If a non-throwing exception specification is
present, an exception propagating from the implicit call to the
`sycl_kernel_launch` template will result in a call to `std::terminate()`.
Otherwise, such an exception will propagate normally.

Functions declared with the `sycl_kernel_entry_point` attribute are not
limited to the simple example shown above. They may have additional template
parameters, declare additional function parameters, and have complex control
flow in the function body. The function must abide by the language feature
restrictions described in section 5.4, "Language restrictions for device
functions" in the SYCL 2020 specification. If the function is a non-static
member function, `this` shall not be used in a potentially evaluated
expression.


### target

{clang-attr-syntaxes}`TargetDocs`

Clang supports the GNU style `__attribute__((target("OPTIONS")))` attribute.
This attribute may be attached to a function definition and instructs
the backend to use different code generation options than were passed on the
command line.

The current set of options correspond to the existing "subtarget features" for
the target with or without a "-mno-" in front corresponding to the absence
of the feature, as well as `arch="CPU"` which will change the default "CPU"
for the function.

For X86, the attribute also allows `tune="CPU"` to optimize the generated
code for the given CPU without changing the available instructions.

For AArch64, `arch="Arch"` will set the architecture, similar to the -march
command line options. `cpu="CPU"` can be used to select a specific cpu,
as per the `-mcpu` option, similarly for `tune=`. The attribute also allows the
`branch-protection=<args>` option, where the permissible arguments and their
effect on code generation are the same as for the command-line option
`-mbranch-protection`.

Example "subtarget features" from the x86 backend include: "mmx", "sse", "sse4.2",
"avx", "xop" and largely correspond to the machine specific options handled by
the front end.

Note that this attribute does not apply transitively to nested functions such
as blocks or C++ lambdas.

Additionally, this attribute supports function multiversioning for ELF based
x86/x86-64 targets, which can be used to create multiple implementations of the
same function that will be resolved at runtime based on the priority of their
`target` attribute strings. A function is considered a multiversioned function
if either two declarations of the function have different `target` attribute
strings, or if it has a `target` attribute string of `default`. For
example:

```c++
__attribute__((target("arch=atom")))
void foo() {} // will be called on 'atom' processors.
__attribute__((target("default")))
void foo() {} // will be called on any other processors.
```

All multiversioned functions must contain a `default` (fallback)
implementation, otherwise usages of the function are considered invalid.
Additionally, a function may not become multiversioned after its first use.


### target_clones

{clang-attr-syntaxes}`TargetClonesDocs`

Clang supports the `target_clones("OPTIONS")` attribute. This attribute may be
attached to a function declaration and causes function multiversioning, where
multiple versions of the function will be emitted with different code
generation options. Additionally, these versions will be resolved at runtime
based on the priority of their attribute options. All `target_clone` functions
are considered multiversioned functions.

For AArch64 target:
The attribute contains comma-separated strings of target features joined by "+"
sign. For example:

```c++
__attribute__((target_clones("sha2+memtag", "fcma+sve2-pmull128")))
void foo() {}
```

For every multiversioned function a `default` (fallback) implementation
always generated if not specified directly.

For x86/x86-64 targets:
All multiversioned functions must contain a `default` (fallback)
implementation, otherwise usages of the function are considered invalid.
Additionally, a function may not become multiversioned after its first use.

The options to `target_clones` can either be a target-specific architecture
(specified as `arch=CPU`), or one of a list of subtarget features.

Example "subtarget features" from the x86 backend include: "mmx", "sse", "sse4.2",
"avx", "xop" and largely correspond to the machine specific options handled by
the front end.

The versions can either be listed as a comma-separated sequence of string
literals or as a single string literal containing a comma-separated list of
versions. For compatibility with GCC, the two formats can be mixed. For
example, the following will emit 4 versions of the function:

```c++
__attribute__((target_clones("arch=atom,avx2","arch=ivybridge","default")))
void foo() {}
```

For targets that support the GNU indirect function (IFUNC) feature, dispatch
is performed by emitting an indirect function that is resolved to the appropriate
target clone at load time. The indirect function is given the name the
multiversioned function would have if it had been declared without the attribute.
For backward compatibility with earlier Clang releases, a function alias with an
`.ifunc` suffix is also emitted. The `.ifunc` suffixed symbol is a deprecated
feature and support for it may be removed in the future.

For PowerPC targets, `target_clones` is supported on AIX only. The attribute
contains comma-separated strings of one of:
(a) `default`, (b) `cpu=CPU`, (c) `FEATURE` or `no-FEATURE`.
The minimum CPU supported is `pwr7` (long spelling such as `power7` is accepted).
The list of target features is a subset of what's allowed on `target`, limited
to what is detectable at runtime using `__builtin_cpu_supports`. IFUNC is supported
on AIX in Clang, so dispatch is implemented similar to other targets using IFUNC.
An FMV function that is only declared in a translation unit is treated as a
non-FMV. The resolver and the function clones are given internal linkage.


### target_version

{clang-attr-syntaxes}`TargetVersionDocs`

For AArch64 target clang supports function multiversioning by
`__attribute__((target_version("OPTIONS")))` attribute. When applied to a
function it instructs compiler to emit multiple function versions based on
`target_version` attribute strings, which resolved at runtime depend on their
priority and target features availability. One of the versions is always
(implicitly or explicitly) the `default` (fallback). Attribute strings can
contain dependent features names joined by the "+" sign.

For targets that support the GNU indirect function (IFUNC) feature, dispatch
is performed by emitting an indirect function that is resolved to the appropriate
target clone at load time. The indirect function is given the name the
multiversioned function would have if it had been declared without the attribute.
For backward compatibility with earlier Clang releases, a function alias with an
`.ifunc` suffix is also emitted. The `.ifunc` suffixed symbol is a deprecated
feature and support for it may be removed in the future.


### try_acquire_capability, try_acquire_shared_capability

{clang-attr-syntaxes}`TryAcquireCapabilityDocs`

Marks a function that attempts to acquire a capability. This function may fail to
actually acquire the capability; they accept a Boolean value determining
whether acquiring the capability means success (true), or failing to acquire
the capability means success (false).


### unsafe_buffer_usage

{clang-attr-syntaxes}`UnsafeBufferUsageDocs`

The attribute `[[clang::unsafe_buffer_usage]]` should be placed on functions
that need to be avoided as they are prone to buffer overflows or unsafe buffer
struct fields. It is designed to work together with the off-by-default compiler
warning `-Wunsafe-buffer-usage` to help codebases transition away from raw pointer
based buffer management, in favor of safer abstractions such as C++20 `std::span`.
The attribute causes `-Wunsafe-buffer-usage` to warn on every use of the function or
the field it is attached to, and it may also lead to emission of automatic fix-it
hints which would help the user replace the use of unsafe functions(/fields) with safe
alternatives, though the attribute can be used even when the fix can't be automated.

- Attribute attached to functions: The attribute suppresses all
  `-Wunsafe-buffer-usage` warnings within the function it is attached to, as the
  function is now classified as unsafe. The attribute should be used carefully, as it
  will silence all unsafe operation warnings inside the function; including any new
  unsafe operations introduced in the future.

  The attribute is warranted even if the only way a function can overflow
  the buffer is by violating the function's preconditions. For example, it
  would make sense to put the attribute on function `foo()` below because
  passing an incorrect size parameter would cause a buffer overflow:

  ```c++
  [[clang::unsafe_buffer_usage]]
  void foo(int *buf, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      buf[i] = i;
    }
  }
  ```

  The attribute is NOT warranted when the function uses safe abstractions,
  assuming that these abstractions weren't misused outside the function.
  For example, function `bar()` below doesn't need the attribute,
  because assuming that the container `buf` is well-formed (has size that
  fits the original buffer it refers to), overflow cannot occur:

  ```c++
  void bar(std::span<int> buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
      buf[i] = i;
    }
  }
  ```

  In this case function `bar()` enables the user to keep the buffer
  "containerized" in a span for as long as possible. On the other hand,
  Function `foo()` in the previous example may have internal
  consistency, but by accepting a raw buffer it requires the user to unwrap
  their span, which is undesirable according to the programming model
  behind `-Wunsafe-buffer-usage`.

  The attribute is warranted when a function accepts a raw buffer only to
  immediately put it into a span:

  ```c++
  [[clang::unsafe_buffer_usage]]
  void baz(int *buf, size_t size) {
    std::span<int> sp{ buf, size };
    for (size_t i = 0; i < sp.size(); ++i) {
      sp[i] = i;
    }
  }
  ```

  In this case `baz()` does not contain any unsafe operations, but the awkward
  parameter type causes the caller to unwrap the span unnecessarily.
  Note that regardless of the attribute, code inside `baz()` isn't flagged
  by `-Wunsafe-buffer-usage` as unsafe. It is definitely undesirable,
  but if `baz()` is on an API surface, there is no way to improve it
  to make it as safe as `bar()` without breaking the source and binary
  compatibility with existing users of the function. In such cases
  the proper solution would be to create a different function (possibly
  an overload of `baz()`) that accepts a safe container like `bar()`,
  and then use the attribute on the original `baz()` to help the users
  update their code to use the new function.

- Attribute attached to fields: The attribute should only be attached to
  struct fields, if the fields can not be updated to a safe type with bounds
  check, such as `std::span`. In other words, the buffers prone to unsafe accesses
  should always be updated to use safe containers/views and attaching the attribute
  must be last resort when such an update is infeasible.

  The attribute can be placed on individual fields or a set of them as shown below.

  ```c++
  struct A {
    [[clang::unsafe_buffer_usage]]
    int *ptr1;

    [[clang::unsafe_buffer_usage]]
    int *ptr2, buf[10];

    [[clang::unsafe_buffer_usage]]
    size_t sz;
  };
  ```

  Here, every read/write to the fields `ptr1`, `ptr2`, `buf` and `sz` will trigger a warning
  that the field has been explicitly marked as unsafe due to unsafe-buffer operations.

- Attribute attached to container constructors and factory functions: The
  spellings `[[clang::unsafe_buffer_usage_in_container]]` and
  `[[clang::unsafe_buffer_usage("container")]]` are equivalent and can be
  placed on two-parameter constructors and factory functions of container or
  view types (such as custom span types taking a `(pointer, size)` or
  `(begin, end)` pair) to opt them in to `-Wunsafe-buffer-usage-in-container`.

  Unlike the general `[[clang::unsafe_buffer_usage]]` attribute, which warns on
  every call, this form suppresses the warning when the argument pair is
  provably safe -- for example, when constructing from `c.data(), c.size()` or
  `c.begin(), c.end()` on the same container object `c`, a constant-sized array
  with a matching bound, `&var, 1`, or a `0` size:

  ```c++
  template <typename T>
  class CustomSpan {
  public:
    [[clang::unsafe_buffer_usage_in_container]]
    CustomSpan(T *ptr, size_t size);

    template <typename It>
    [[clang::unsafe_buffer_usage("container")]]
    CustomSpan(It first, It last);
  };

  template <typename T>
  [[clang::unsafe_buffer_usage("container")]]
  CustomSpan<T> MakeCustomSpan(T *ptr, size_t size);

  void example(int *p, size_t n, MyVector<int> &v) {
    CustomSpan<int> s1(p, n);                     // warning: decoupled pointer and size
    auto s2 = MakeCustomSpan(p, n);               // warning: decoupled pointer and size
    CustomSpan<int> s3(v.data(), v.size());       // no warning
    CustomSpan<int> s4(v.begin(), v.end());       // no warning
    auto s5 = MakeCustomSpan(v.data(), v.size()); // no warning
  }
  ```


### used

{clang-attr-syntaxes}`UsedDocs`

This attribute, when attached to a function or variable definition, indicates
that there may be references to the entity which are not apparent in the source
code. For example, it may be referenced from inline `asm`, or it may be
found through a dynamic symbol or section lookup.

The compiler must emit the definition even if it appears to be unused, and it
must not apply optimizations which depend on fully understanding how the entity
is used.

Whether this attribute has any effect on the linker depends on the target and
the linker. Most linkers support the feature of section garbage collection
(`--gc-sections`), also known as "dead stripping" (`ld64 -dead_strip`) or
discarding unreferenced sections (`link.exe /OPT:REF`). On COFF and Mach-O
targets (Windows and Apple platforms), the `used` attribute prevents symbols
from being removed by linker section GC. On ELF targets, it has no effect on its
own, and the linker may remove the definition if it is not otherwise referenced.
This linker GC can be avoided by also adding the `retain` attribute. Note
that `retain` requires special support from the linker; see that attribute's
documentation for further information.


### xray_always_instrument, xray_never_instrument, xray_log_args

{clang-attr-syntaxes}`XRayDocs`

`__attribute__((xray_always_instrument))` or
`[[clang::xray_always_instrument]]` is used to mark member functions (in C++),
methods (in Objective C), and free functions (in C, C++, and Objective C) to be
instrumented with XRay. This will cause the function to always have space at
the beginning and exit points to allow for runtime patching.

Conversely, `__attribute__((xray_never_instrument))` or
`[[clang::xray_never_instrument]]` will inhibit the insertion of these
instrumentation points.

If a function has neither of these attributes, they become subject to the XRay
heuristics used to determine whether a function should be instrumented or
otherwise.

`__attribute__((xray_log_args(N)))` or `[[clang::xray_log_args(N)]]` is
used to preserve N function arguments for the logging function. Currently,
only N==1 is supported.


### zero_call_used_regs

{clang-attr-syntaxes}`ZeroCallUsedRegsDocs`

This attribute, when attached to a function, causes the compiler to zero a
subset of all call-used registers before the function returns. It's used to
increase program security by either mitigating [Return-Oriented Programming][return-oriented programming]
(ROP) attacks or preventing information leakage through registers.

The term "call-used" means registers which are not guaranteed to be preserved
unchanged for the caller by the current calling convention. This could also be
described as "caller-saved" or "not callee-saved".

The `choice` parameters gives the programmer flexibility to choose the subset
of the call-used registers to be zeroed:

- `skip` doesn't zero any call-used registers. This choice overrides any
  command-line arguments.
- `used` only zeros call-used registers used in the function. By `used`, we
  mean a register whose contents have been set or referenced in the function.
- `used-gpr` only zeros call-used GPR registers used in the function.
- `used-arg` only zeros call-used registers used to pass arguments to the
  function.
- `used-gpr-arg` only zeros call-used GPR registers used to pass arguments to
  the function.
- `all` zeros all call-used registers.
- `all-gpr` zeros all call-used GPR registers.
- `all-arg` zeros all call-used registers used to pass arguments to the
  function.
- `all-gpr-arg` zeros all call-used GPR registers used to pass arguments to
  the function.

The default for the attribute is controlled by the `-fzero-call-used-regs`
flag.

[return-oriented programming]: https://en.wikipedia.org/wiki/Return-oriented_programming


