## Declaration Attributes



### Owner

{clang-attr-syntaxes}`LifetimeOwnerDocs`

:::{Note}
This attribute is experimental and its effect on analysis is subject to change in
a future version of clang.
:::

The attribute `[[gsl::Owner(T)]]` applies to structs and classes that own an
object of type `T`:

```
class [[gsl::Owner(int)]] IntOwner {
private:
  int value;
public:
  int *getInt() { return &value; }
};
```

The argument `T` is optional and is ignored.
This attribute may be used by analysis tools and has no effect on code
generation. A `void` argument means that the class can own any type.

See [Pointer](#pointer) for an example.


### Pointer

{clang-attr-syntaxes}`LifetimePointerDocs`

:::{Note}
This attribute is experimental and its effect on analysis is subject to change in
a future version of clang.
:::

The attribute `[[gsl::Pointer(T)]]` applies to structs and classes that behave
like pointers to an object of type `T`:

```
class [[gsl::Pointer(int)]] IntPointer {
private:
  int *valuePointer;
public:
  IntPointer(const IntOwner&);
  int *getInt() { return valuePointer; }
};
```

The argument `T` is optional and is ignored.
This attribute may be used by analysis tools and has no effect on code
generation. A `void` argument means that the pointer can point to any type.

Example:
When constructing an instance of a class annotated like this (a Pointer) from
an instance of a class annotated with `[[gsl::Owner]]` (an Owner),
then the analysis will consider the Pointer to point inside the Owner.
When the Owner's lifetime ends, it will consider the Pointer to be dangling.

```c++
int f() {
  IntPointer P(IntOwner{}); // P "points into" a temporary IntOwner object
  P.getInt(); // P is dangling
}
```

**Transparent Member Functions**

The analysis automatically tracks certain member functions of `[[gsl::Pointer]]` types
that provide transparent access to the pointed-to object. These include:

- Dereference operators: `operator*`, `operator->`
- Data access methods: `data()`, `c_str()`, `get()`
- Iterator operations: `begin()`, `end()`, `rbegin()`, `rend()`, `cbegin()`, `cend()`, `crbegin()`, `crend()`, `operator+`, `operator-`, `operator++`, `operator--`

When these methods return pointers, view types, or references, the analysis treats them as
transparently borrowing from the same object that the pointer itself borrows from,
enabling detection of use-after-free through these access patterns:

```c++
// For example, .data() here returns a borrow to 's' instead of 'v'.
const char* f() {
  std::string s = "hello";
  std::string_view v = s; // warning: address of stack memory returned
  return v.data();        // note: returned here
}

const MyObj& g(MyObj obj) {
  View v = obj; // warning: address of stack memory returned
  return *v;    // note: returned here
}
```

This tracking also applies to range-based for loops, where the `begin()` and `end()`
iterators are used to access elements:

```c++
std::string_view f(std::vector<std::string> vec) {
  for (const std::string& s : vec) {  // warning: address of stack memory returned
    return s; // note: returned here
  }
}
```

**Container Template Specialization**

If a template class is annotated with `[[gsl::Owner]]`, and the first
instantiated template argument is a pointer type (raw pointer, or `[[gsl::Pointer]]`),
the analysis will consider the instantiated class as a container of the pointer.
When constructing such an object from a GSL owner object, the analysis will
assume that the container holds a pointer to the owner object. Consequently,
when the owner object is destroyed, the pointer will be considered dangling.

```c++
int f() {
  std::vector<std::string_view> v = {std::string()}; // v holds a dangling pointer.
  std::optional<std::string_view> o = std::string(); // o holds a dangling pointer.
}
```


### __single_inheritance, __multiple_inheritance, __virtual_inheritance

{clang-attr-syntaxes}`MSInheritanceDocs`

This collection of keywords is enabled under `-fms-extensions` and controls
the pointer-to-member representation used on `*-*-win32` targets.

The `*-*-win32` targets utilize a pointer-to-member representation which
varies in size and alignment depending on the definition of the underlying
class.

However, this is problematic when a forward declaration is only available and
no definition has been made yet. In such cases, Clang is forced to utilize the
most general representation that is available to it.

These keywords make it possible to use a pointer-to-member representation other
than the most general one regardless of whether or not the definition will ever
be present in the current translation unit.

This family of keywords belong between the `class-key` and `class-name`:

```c++
struct __single_inheritance S;
int S::*i;
struct S {};
```

This keyword can be applied to class templates but only has an effect when used
on full specializations:

```c++
template <typename T, typename U> struct __single_inheritance A; // warning: inheritance model ignored on primary template
template <typename T> struct __multiple_inheritance A<T, T>; // warning: inheritance model ignored on partial specialization
template <> struct __single_inheritance A<int, float>;
```

Note that choosing an inheritance model less general than strictly necessary is
an error:

```c++
struct __multiple_inheritance S; // error: inheritance model does not match definition
int S::*i;
struct S {};
```


### annotate

{clang-attr-syntaxes}`AnnotateDocs`

The `annotate` attribute is used to add annotations to declarations or statements,
typically for use by static analysis tools that are not integrated into the
core Clang compiler (e.g., Clang-Tidy checks or out-of-tree Clang-based tools).
It is a counterpart to the `annotate_type` attribute, which serves the same
purpose, but for types.

The attribute takes a mandatory string literal argument specifying the
annotation category and an arbitrary number of optional arguments that provide
additional information specific to the annotation category. The optional
arguments must be constant expressions of arbitrary type.

For example:

```c++
[[clang::annotate("category1", "foo", 1)]] void func(int val [[clang::annotate("category2")]]) {
  [[clang::annotate("category3")]] if (val) {

  }
}
```


### asm

{clang-attr-syntaxes}`AsmLabelDocs`

This attribute can be used on a function or variable to specify its symbol name.

On some targets, all C symbols are prefixed by default with a single character,
typically `_`. This was done historically to distinguish them from symbols
used by other languages. (This prefix is also added to the standard Itanium
C++ ABI prefix on "mangled" symbol names, so that e.g. on such targets the true
symbol name for a C++ variable declared as `int cppvar;` would be
`__Z6cppvar`; note the two underscores.) This prefix is *not* added to the
symbol names specified by the `__asm` attribute; programmers wishing to match
a C symbol name must compensate for this.

For example, consider the following C code:

```c
int var1 __asm("altvar") = 1;  // "altvar" in symbol table.
int var2 = 1; // "_var2" in symbol table.

void func1(void) __asm("altfunc");
void func1(void) {} // "altfunc" in symbol table.
void func2(void) {} // "_func2" in symbol table.
```

Clang's implementation of this attribute is compatible with GCC's, [documented here](https://gcc.gnu.org/onlinedocs/gcc/Asm-Labels.html).

While it is possible to use this attribute to name a special symbol used
internally by the compiler, such as an LLVM intrinsic, this is neither
recommended nor supported and may cause the compiler to crash or miscompile.
Users who wish to gain access to intrinsic behavior are strongly encouraged to
request new builtin functions.


### cluster_dims

{clang-attr-syntaxes}`CUDAClusterDimsAttrDoc`

In CUDA/HIP programming, the `cluster_dims` attribute, conventionally exposed as the
`__cluster_dims__` macro, can be applied to a kernel function to set the dimensions of a
thread block cluster, which is an optional level of hierarchy and made up of thread blocks.
`__cluster_dims__` defines the cluster size as `(X, Y, Z)`, where each value is the number
of thread blocks in that dimension. The `cluster_dims` and `no_cluster` attributes are
mutually exclusive.

```
__global__ __cluster_dims__(2, 1, 1) void kernel(...) {
  ...
}
```


### coro_await_elidable

{clang-attr-syntaxes}`CoroAwaitElidableDoc`

The `[[clang::coro_await_elidable]]` is a class attribute which can be
applied to a coroutine return type. It provides a hint to the compiler to apply
Heap Allocation Elision more aggressively.

When a coroutine function returns such a type, a direct call expression therein
that returns a prvalue of a type attributed `[[clang::coro_await_elidable]]`
is said to be under a safe elide context if one of the following is true:

- it is the immediate right-hand side operand to a co_await expression.
- it is an argument to a `[[clang::coro_await_elidable_argument]]` parameter
  or parameter pack of another direct call expression under a safe elide context.

Do note that the safe elide context applies only to the call expression itself,
and the context does not transitively include any of its subexpressions unless
exceptional rules of `[[clang::coro_await_elidable_argument]]` apply.

The compiler performs heap allocation elision on call expressions under a safe
elide context, if the callee is a coroutine.

Example:

```c++
class [[clang::coro_await_elidable]] Task { ... };

Task foo();
Task bar() {
  co_await foo(); // foo()'s coroutine frame on this line is elidable
  auto t = foo(); // foo()'s coroutine frame on this line is NOT elidable
  co_await t;
}
```

Such elision replaces the heap allocated activation frame of the callee coroutine
with a local variable within the enclosing braces in the caller's stack frame.
The local variable, like other variables in coroutines, may be collected into the
coroutine frame, which may be allocated on the heap. The behavior is undefined
if the caller coroutine is destroyed earlier than the callee coroutine.


### coro_await_elidable_argument

{clang-attr-syntaxes}`CoroAwaitElidableArgumentDoc`

The `[[clang::coro_await_elidable_argument]]` is a function parameter attribute.
It works in conjunction with `[[clang::coro_await_elidable]]` to propagate a
safe elide context to a parameter or parameter pack if the function is called
under a safe elide context.

This is sometimes necessary on utility functions used to compose or modify the
behavior of a callee coroutine.

Example:

```c++
template <typename T>
class [[clang::coro_await_elidable]] Task { ... };

template <typename... T>
class [[clang::coro_await_elidable]] WhenAll { ... };

// `when_all` is a utility function that composes coroutines. It does not
// need to be a coroutine to propagate.
template <typename... T>
WhenAll<T...> when_all([[clang::coro_await_elidable_argument]] Task<T> tasks...);

Task<int> foo();
Task<int> bar();
Task<void> example1() {
  // `when_all`, `foo`, and `bar` are all elide safe because `when_all` is
  // under a safe elide context and, thanks to the [[clang::coro_await_elidable_argument]]
  // attribute, such context is propagated to foo and bar.
  co_await when_all(foo(), bar());
}

Task<void> example2() {
  // `when_all` and `bar` are elide safe. `foo` is not elide safe.
  auto f = foo();
  co_await when_all(f, bar());
}


Task<void> example3() {
  // None of the calls are elide safe.
  auto t = when_all(foo(), bar());
  co_await t;
}
```


### coro_disable_lifetimebound, coro_lifetimebound

{clang-attr-syntaxes}`CoroLifetimeBoundDoc`

The `[[clang::coro_lifetimebound]]` is a class attribute which can be applied
to a coroutine return type ([coro_return_type, coro_wrapper]) (i.e.
it should also be annotated with `[[clang::coro_return_type]]`).

All parameters of a function are considered to be lifetime bound if the function returns a
coroutine return type (CRT) annotated with `[[clang::coro_lifetimebound]]`.
This lifetime bound analysis can be disabled for a coroutine wrapper or a coroutine by annotating the function
with `[[clang::coro_disable_lifetimebound]]` function attribute .
See documentation of [lifetimebound] for details about lifetime bound analysis.

Reference parameters of a coroutine are susceptible to capturing references to temporaries or local variables.

For example,

```c++
task<int> coro(const int& a) { co_return a + 1; }
task<int> dangling_refs(int a) {
  // `coro` captures reference to a temporary. `foo` would now contain a dangling reference to `a`.
  auto foo = coro(1);
  // `coro` captures reference to local variable `a` which is destroyed after the return.
  return coro(a);
}
```

Lifetime bound static analysis can be used to detect such instances when coroutines capture references
which may die earlier than the coroutine frame itself. In the above example, if the CRT `task` is annotated with
`[[clang::coro_lifetimebound]]`, then lifetime bound analysis would detect capturing reference to
temporaries or return address of a local variable.

Both coroutines and coroutine wrappers are part of this analysis.

```c++
template <typename T> struct [[clang::coro_return_type, clang::coro_lifetimebound]] Task {
  using promise_type = some_promise_type;
};

Task<int> coro(const int& a) { co_return a + 1; }
[[clang::coro_wrapper]] Task<int> coro_wrapper(const int& a, const int& b) {
  return a > b ? coro(a) : coro(b);
}
Task<int> temporary_reference() {
  auto foo = coro(1); // warning: capturing reference to a temporary which would die after the expression.

  int a = 1;
  auto bar = coro_wrapper(a, 0); // warning: `b` captures reference to a temporary.

  co_return co_await coro(1); // fine.
}
[[clang::coro_wrapper]] Task<int> stack_reference(int a) {
  return coro(a); // warning: returning address of stack variable `a`.
}
```

This analysis can be disabled for all calls to a particular function by annotating the function
with function attribute `[[clang::coro_disable_lifetimebound]]`.
For example, this could be useful for coroutine wrappers which accept reference parameters
but do not pass them to the underlying coroutine or pass them by value.

```c++
Task<int> coro(int a) { co_return a + 1; }
[[clang::coro_wrapper, clang::coro_disable_lifetimebound]] Task<int> coro_wrapper(const int& a) {
  return coro(a + 1);
}
void use() {
  auto task = coro_wrapper(1); // use of temporary is fine as the argument is not lifetime bound.
}
```


### coro_only_destroy_when_complete

{clang-attr-syntaxes}`CoroOnlyDestroyWhenCompleteDocs`

The `coro_only_destroy_when_complete` attribute should be marked on a C++ class. The coroutines
whose return type is marked with the attribute are assumed to be destroyed only after the coroutine has
reached the final suspend point.

This is helpful for the optimizers to reduce the size of the destroy function for the coroutines.

For example,

```c++
A foo() {
  dtor d;
  co_await something();
  dtor d1;
  co_await something();
  dtor d2;
  co_return 43;
}
```

The compiler may generate the following pseudocode:

```c++
void foo.destroy(foo.Frame *frame) {
  switch(frame->suspend_index()) {
    case 1:
      frame->d.~dtor();
      break;
    case 2:
      frame->d.~dtor();
      frame->d1.~dtor();
      break;
    case 3:
      frame->d.~dtor();
      frame->d1.~dtor();
      frame->d2.~dtor();
      break;
    default: // coroutine completed or haven't started
      break;
  }

  frame->promise.~promise_type();
  delete frame;
}
```

The `foo.destroy()` function's purpose is to release all of the resources
initialized for the coroutine when it is destroyed in a suspended state.
However, if the coroutine is only ever destroyed at the final suspend state,
the rest of the conditions are superfluous.

The user can use the `coro_only_destroy_when_complete` attributo suppress
generation of the other destruction cases, optimizing the above `foo.destroy` to:

```c++
void foo.destroy(foo.Frame *frame) {
  frame->promise.~promise_type();
  delete frame;
}
```


### coro_return_type, coro_wrapper

{clang-attr-syntaxes}`CoroReturnTypeAndWrapperDoc`

The `[[clang::coro_return_type]]` attribute is used to help static analyzers to recognize
coroutines from the function signatures.

The `coro_return_type` attribute should be marked on a C++ class to mark it as
a **coroutine return type (CRT)**.

A function `R func(P1, .., PN)` has a coroutine return type (CRT) `R` if `R`
is marked by `[[clang::coro_return_type]]` and `R` has a promise type associated to it
(i.e., `std::coroutine_traits<R, P1, .., PN>::promise_type` is a valid promise type).

If the return type of a function is a `CRT` then the function must be a coroutine.
Otherwise the program is invalid. It is allowed for a non-coroutine to return a `CRT`
if the function is marked with `[[clang::coro_wrapper]]`.

The `[[clang::coro_wrapper]]` attribute should be marked on a C++ function to mark it as
a **coroutine wrapper**. A coroutine wrapper is a function which returns a `CRT`,
is not a coroutine itself and is marked with `[[clang::coro_wrapper]]`.

Clang will enforce that all functions that return a `CRT` are either coroutines or marked
with `[[clang::coro_wrapper]]`. Clang will enforce this with an error.

From a language perspective, it is not possible to differentiate between a coroutine and a
function returning a CRT by merely looking at the function signature.

Coroutine wrappers, in particular, are susceptible to capturing
references to temporaries and other lifetime issues. This allows to avoid such lifetime
issues with coroutine wrappers.

For example,

```c++
// This is a CRT.
template <typename T> struct [[clang::coro_return_type]] Task {
  using promise_type = some_promise_type;
};

Task<int> increment(int a) { co_return a + 1; } // Fine. This is a coroutine.
Task<int> foo() { return increment(1); } // Error. foo is not a coroutine.

// Fine for a coroutine wrapper to return a CRT.
[[clang::coro_wrapper]] Task<int> foo() { return increment(1); }

void bar() {
  // Invalid. This intantiates a function which returns a CRT but is not marked as
  // a coroutine wrapper.
  std::function<Task<int>(int)> f = increment;
}
```

Note: `a_promise_type::get_return_object` is exempted from this analysis as it is a necessary
implementation detail of any coroutine library.


### deprecated

{clang-attr-syntaxes}`DeprecatedDocs`

The `deprecated` attribute can be applied to a function, a variable, or a
type. This is useful when identifying functions, variables, or types that are
expected to be removed in a future version of a program.

Consider the function declaration for a hypothetical function `f`:

```c++
void f(void) __attribute__((deprecated("message", "replacement")));
```

When spelled as `__attribute__((deprecated))`, the deprecated attribute can have
two optional string arguments. The first one is the message to display when
emitting the warning; the second one enables the compiler to provide a Fix-It
to replace the deprecated name with a new name. Otherwise, when spelled as
`[[gnu::deprecated]]` or `[[deprecated]]`, the attribute can have one optional
string argument which is the message to display when emitting the warning.


### empty_bases

{clang-attr-syntaxes}`EmptyBasesDocs`

The empty_bases attribute permits the compiler to utilize the
empty-base-optimization more frequently.
This attribute only applies to struct, class, and union types.
It is only supported when using the Microsoft C++ ABI.


### enum_extensibility

{clang-attr-syntaxes}`EnumExtensibilityDocs`

Attribute `enum_extensibility` is used to distinguish between enum definitions
that are extensible and those that are not. The attribute can take either
`closed` or `open` as an argument. `closed` indicates a variable of the
enum type takes a value that corresponds to one of the enumerators listed in the
enum definition or, when the enum is annotated with `flag_enum`, a value that
can be constructed using values corresponding to the enumerators. `open`
indicates a variable of the enum type can take any values allowed by the
standard and instructs clang to be more lenient when issuing warnings.

```c
enum __attribute__((enum_extensibility(closed))) ClosedEnum {
  A0, A1
};

enum __attribute__((enum_extensibility(open))) OpenEnum {
  B0, B1
};

enum __attribute__((enum_extensibility(closed),flag_enum)) ClosedFlagEnum {
  C0 = 1 << 0, C1 = 1 << 1
};

enum __attribute__((enum_extensibility(open),flag_enum)) OpenFlagEnum {
  D0 = 1 << 0, D1 = 1 << 1
};

void foo1() {
  enum ClosedEnum ce;
  enum OpenEnum oe;
  enum ClosedFlagEnum cfe;
  enum OpenFlagEnum ofe;

  ce = A1;           // no warnings
  ce = 100;          // warning issued
  oe = B1;           // no warnings
  oe = 100;          // no warnings
  cfe = C0 | C1;     // no warnings
  cfe = C0 | C1 | 4; // warning issued
  ofe = D0 | D1;     // no warnings
  ofe = D0 | D1 | 4; // no warnings
}
```


### external_source_symbol

{clang-attr-syntaxes}`ExternalSourceSymbolDocs`

The `external_source_symbol` attribute specifies that a declaration originates
from an external source and describes the nature of that source.

The fact that Clang is capable of recognizing declarations that were defined
externally can be used to provide better tooling support for mixed-language
projects or projects that rely on auto-generated code. For instance, an IDE that
uses Clang and that supports mixed-language projects can use this attribute to
provide a correct "jump-to-definition" feature. For a concrete example,
consider a protocol that's defined in a Swift file:

```swift
@objc public protocol SwiftProtocol {
  func method()
}
```

This protocol can be used from Objective-C code by including a header file that
was generated by the Swift compiler. The declarations in that header can use
the `external_source_symbol` attribute to make Clang aware of the fact
that `SwiftProtocol` actually originates from a Swift module:

```objc
__attribute__((external_source_symbol(language="Swift",defined_in="module")))
@protocol SwiftProtocol
@required
- (void) method;
@end
```

Consequently, when "jump-to-definition" is performed at a location that
references `SwiftProtocol`, the IDE can jump to the original definition in
the Swift source file rather than jumping to the Objective-C declaration in the
auto-generated header file.

The `external_source_symbol` attribute is a comma-separated list that includes
clauses that describe the origin and the nature of the particular declaration.
Those clauses can be:

language=*string-literal*

: The name of the source language in which this declaration was defined.

defined_in=*string-literal*

: The name of the source container in which the declaration was defined. The
  exact definition of source container is language-specific, e.g. Swift's
  source containers are modules, so `defined_in` should specify the Swift
  module name.

USR=*string-literal*

: String that specifies a unified symbol resolution (USR) value for this
  declaration. USR string uniquely identifies this particular declaration, and
  is typically used when constructing an index of a codebase.
  The USR value in this attribute is expected to be generated by an external
  compiler that compiled the native declaration using its original source
  language. The exact format of the USR string and its other attributes
  are determined by the specification of this declaration's source language.
  When not specified, Clang's indexer will use the Clang USR for this symbol.
  User can query to see if Clang supports the use of the `USR` clause in
  the `external_source_symbol` attribute with
  `__has_attribute(external_source_symbol) >= 20230206`.

generated_declaration

: This declaration was automatically generated by some tool.

The clauses can be specified in any order. The clauses that are listed above are
all optional, but the attribute has to have at least one clause.


### flag_enum

{clang-attr-syntaxes}`FlagEnumDocs`

This attribute can be added to an enumerator to signal to the compiler that it
is intended to be used as a flag type. This will cause the compiler to assume
that the range of the type includes all of the values that you can get by
manipulating bits of the enumerator when issuing warnings.


### gcc_struct, ms_struct

{clang-attr-syntaxes}`MSStructDocs`

The `ms_struct` and `gcc_struct` attributes request the compiler to enter a
special record layout compatibility mode which mimics the layout of Microsoft or
Itanium C++ ABI respectively. Obviously, if the current C++ ABI matches the
requested ABI, the attribute does nothing. However, if it does not, annotated
structure or class is laid out in a special compatibility mode, which slightly
changes offsets for fields and bit-fields. The intention is to match the layout
of the requested ABI for structures which only use C features.

Note that the default behavior can be controlled by `-mms-bitfields` and
`-mno-ms-bitfields` switches and via `#pragma ms_struct`.

The primary difference is for bitfields, where the MS variant only packs
adjacent fields into the same allocation unit if they have integral types
of the same size, while the GCC/Itanium variant packs all fields in a bitfield
tightly.


### grid_constant

{clang-attr-syntaxes}`CUDAGridConstantAttrDocs`

The `__grid_constant__` attribute can be applied to a `const`-qualified kernel
function argument and allows compiler to take the address of that argument without
making a copy. The argument applies to sm_70 or newer GPUs, during compilation
with CUDA-11.7(PTX 7.7) or newer, and is ignored otherwise.


### layout_version

{clang-attr-syntaxes}`LayoutVersionDocs`

The layout_version attribute requests that the compiler utilize the class
layout rules of a particular compiler version.
This attribute only applies to struct, class, and union types.
It is only supported when using the Microsoft C++ ABI.


### lto_visibility_public

{clang-attr-syntaxes}`LTOVisibilityDocs`

See {doc}`LTOVisibility`.


### managed

{clang-attr-syntaxes}`HIPManagedAttrDocs`

The `__managed__` attribute can be applied to a global variable declaration in HIP.
A managed variable is emitted as an undefined global symbol in the device binary and is
registered by `__hipRegisterManagedVar` in init functions. The HIP runtime allocates
managed memory and uses it to define the symbol when loading the device binary.
A managed variable can be accessed in both device and host code.


### no_cluster

{clang-attr-syntaxes}`CUDANoClusterAttrDoc`

In CUDA/HIP programming, a kernel function can still be launched with the cluster feature enabled
at runtime, even without being annotated with `__cluster_dims__`. The LLVM/Clang-exclusive
`no_cluster` attribute, conventionally exposed as the `__no_cluster__` macro, can be applied to
a kernel function to explicitly indicate that the cluster feature will not be enabled either at
compile time or at kernel launch time. This allows the compiler to apply certain optimizations
without assuming that clustering could be enabled at runtime. It is undefined behavior to launch a
kernel annotated with `__no_cluster__` if the cluster feature is enabled at runtime.
The `cluster_dims` and `no_cluster` attributes are mutually exclusive.

```
__global__ __no_cluster__ void kernel(...) {
  ...
}
```


### no_init_all

{clang-attr-syntaxes}`NoTrivialAutoVarInitDocs`

The `__declspec(no_init_all)` attribute disables the automatic initialization
that the {option}`-ftrivial-auto-var-init` flag would have applied to locals in
a marked function, or instances of a marked type. Note that this attribute has
no effect for locals that are automatically initialized without the
{option}`-ftrivial-auto-var-init` flag.


### no_specializations

{clang-attr-syntaxes}`NoSpecializationsDocs`

`[[clang::no_specializations]]` can be applied to function, class, or variable
templates for which neither an explicit specialization nor a partial specialization should be declared by users. This is primarily
used to diagnose user specializations of standard library type traits.


### nonstring

{clang-attr-syntaxes}`NonStringDocs`

The `nonstring` attribute can be applied to the declaration of a variable or
a field whose type is a character pointer or character array to specify that
the buffer is not intended to behave like a null-terminated string. This will
silence diagnostics with code like:

```c
char BadStr[3] = "foo"; // No space for the null terminator, diagnosed
__attribute__((nonstring)) char NotAStr[3] = "foo"; // Not diagnosed
```


### novtable

{clang-attr-syntaxes}`MSNoVTableDocs`

This attribute can be added to a class declaration or definition to signal to
the compiler that constructors and destructors will not reference the virtual
function table. It is only supported when using the Microsoft C++ ABI.


### ns_error_domain

{clang-attr-syntaxes}`NSErrorDomainDocs`

In Cocoa frameworks in Objective-C, one can group related error codes in enums
and categorize these enums with error domains.

The `ns_error_domain` attribute indicates a global `NSString` or
`CFString` constant representing the error domain that an error code belongs
to. For pointer uniqueness and code size this is a constant symbol, not a
literal.

The domain and error code need to be used together. The `ns_error_domain`
attribute links error codes to their domain at the source level.

This metadata is useful for documentation purposes, for static analysis, and for
improving interoperability between Objective-C and Swift. It is not used for
code generation in Objective-C.

For example:

```objc
#define NS_ERROR_ENUM(_type, _name, _domain)  \
  enum _name : _type _name; enum __attribute__((ns_error_domain(_domain))) _name : _type

extern NSString *const MyErrorDomain;
typedef NS_ERROR_ENUM(unsigned char, MyErrorEnum, MyErrorDomain) {
  MyErrFirst,
  MyErrSecond,
};
```


### objc_boxable

{clang-attr-syntaxes}`ObjCBoxableDocs`

Structs and unions marked with the `objc_boxable` attribute can be used
with the Objective-C boxed expression syntax, `@(...)`.

**Usage**: `__attribute__((objc_boxable))`. This attribute
can only be placed on a declaration of a trivially-copyable struct or union:

```objc
struct __attribute__((objc_boxable)) some_struct {
  int i;
};
union __attribute__((objc_boxable)) some_union {
  int i;
  float f;
};
typedef struct __attribute__((objc_boxable)) _some_struct some_struct;

// ...

some_struct ss;
NSValue *boxed = @(ss);
```


### objc_direct

{clang-attr-syntaxes}`ObjCDirectDocs`

The `objc_direct` attribute can be used to mark an Objective-C method as
being *direct*. A direct method is treated statically like an ordinary method,
but dynamically it behaves more like a C function. This lowers some of the costs
associated with the method but also sacrifices some of the ordinary capabilities
of Objective-C methods.

A message send of a direct method calls the implementation directly, as if it
were a C function, rather than using ordinary Objective-C method dispatch. This
is substantially faster and potentially allows the implementation to be inlined,
but it also means the method cannot be overridden in subclasses or replaced
dynamically, as ordinary Objective-C methods can.

Furthermore, a direct method is not listed in the class's method lists. This
substantially reduces the code-size overhead of the method but also means it
cannot be called dynamically using ordinary Objective-C method dispatch at all;
in particular, this means that it cannot override a superclass method or satisfy
a protocol requirement.

Because a direct method cannot be overridden, it is an error to perform
a `super` message send of one.

Although a message send of a direct method causes the method to be called
directly as if it were a C function, it still obeys Objective-C semantics in other
ways:

- If the receiver is `nil`, the message send does nothing and returns the zero value
  for the return type.
- A message send of a direct class method will cause the class to be initialized,
  including calling the `+initialize` method if present.
- The implicit `_cmd` parameter containing the method's selector is still defined.
  In order to minimize code-size costs, the implementation will not emit a reference
  to the selector if the parameter is unused within the method.

Symbols for direct method implementations are implicitly given hidden
visibility, meaning that they can only be called within the same linkage unit.

It is an error to do any of the following:

- declare a direct method in a protocol,
- declare an override of a direct method with a method in a subclass,
- declare an override of a non-direct method with a direct method in a subclass,
- declare a method with different directness in different class interfaces, or
- implement a non-direct method (as declared in any class interface) with a direct method.

If any of these rules would be violated if every method defined in an
`@implementation` within a single linkage unit were declared in an
appropriate class interface, the program is ill-formed with no diagnostic
required. If a violation of this rule is not diagnosed, behavior remains
well-defined; this paragraph is simply reserving the right to diagnose such
conflicts in the future, not to treat them as undefined behavior.

Additionally, Clang will warn about any `@selector` expression that
names a selector that is only known to be used for direct methods.

For the purpose of these rules, a "class interface" includes a class's primary
`@interface` block, its class extensions, its categories, its declared protocols,
and all the class interfaces of its superclasses.

An Objective-C property can be declared with the `direct` property
attribute. If a direct property declaration causes an implicit declaration of
a getter or setter method (that is, if the given method is not explicitly
declared elsewhere), the method is declared to be direct.

Some programmers may wish to make many methods direct at once. In order
to simplify this, the `objc_direct_members` attribute is provided; see its
documentation for more information.


### objc_direct_members

{clang-attr-syntaxes}`ObjCDirectMembersDocs`

The `objc_direct_members` attribute can be placed on an Objective-C
`@interface` or `@implementation` to mark that methods declared
therein should be considered direct by default. See the documentation
for `objc_direct` for more information about direct methods.

When `objc_direct_members` is placed on an `@interface` block, every
method in the block is considered to be declared as direct. This includes any
implicit method declarations introduced by property declarations. If the method
redeclares a non-direct method, the declaration is ill-formed, exactly as if the
method was annotated with the `objc_direct` attribute.

When `objc_direct_members` is placed on an `@implementation` block,
methods defined in the block are considered to be declared as direct unless
they have been previously declared as non-direct in any interface of the class.
This includes the implicit method definitions introduced by synthesized
properties, including auto-synthesized properties.


### objc_non_runtime_protocol

{clang-attr-syntaxes}`ObjCNonRuntimeProtocolDocs`

The `objc_non_runtime_protocol` attribute can be used to mark that an
Objective-C protocol is only used during static type-checking and doesn't need
to be represented dynamically. This avoids several small code-size and run-time
overheads associated with handling the protocol's metadata. A non-runtime
protocol cannot be used as the operand of a `@protocol` expression, and
dynamic attempts to find it with `objc_getProtocol` will fail.

If a non-runtime protocol inherits from any ordinary protocols, classes and
derived protocols that declare conformance to the non-runtime protocol will
dynamically list their conformance to those bare protocols.


### objc_nonlazy_class

{clang-attr-syntaxes}`ObjCNonLazyClassDocs`

This attribute can be added to an Objective-C `@interface` or
`@implementation` declaration to add the class to the list of non-lazily
initialized classes. A non-lazy class will be initialized eagerly when the
Objective-C runtime is loaded. This is required for certain system classes which
have instances allocated in non-standard ways, such as the classes for blocks
and constant strings. Adding this attribute is essentially equivalent to
providing a trivial `+load` method but avoids the (fairly small) load-time
overheads associated with defining and calling such a method.


### objc_runtime_name

{clang-attr-syntaxes}`ObjCRuntimeNameDocs`

By default, the Objective-C interface or protocol identifier is used
in the metadata name for that object. The `objc_runtime_name`
attribute allows annotated interfaces or protocols to use the
specified string argument in the object's metadata name instead of the
default name.

**Usage**: `__attribute__((objc_runtime_name("MyLocalName")))`. This attribute
can only be placed before an @protocol or @interface declaration:

```objc
__attribute__((objc_runtime_name("MyLocalName")))
@interface Message
@end
```


### objc_runtime_visible

{clang-attr-syntaxes}`ObjCRuntimeVisibleDocs`

This attribute specifies that the Objective-C class to which it applies is
visible to the Objective-C runtime but not to the linker. Classes annotated
with this attribute cannot be subclassed and cannot have categories defined for
them.


### objc_subclassing_restricted

{clang-attr-syntaxes}`ObjCSubclassingRestrictedDocs`

This attribute can be added to an Objective-C `@interface` declaration to
ensure that this class cannot be subclassed.


### preferred_name

{clang-attr-syntaxes}`PreferredNameDocs`

The `preferred_name` attribute can be applied to a class template, and
specifies a preferred way of naming a specialization of the template. The
preferred name will be used whenever the corresponding template specialization
would otherwise be printed in a diagnostic or similar context.

The preferred name must be a typedef or type alias declaration that refers to a
specialization of the class template (not including any type qualifiers). In
general this requires the template to be declared at least twice. For example:

```c++
template<typename T> struct basic_string;
using string = basic_string<char>;
using wstring = basic_string<wchar_t>;
template<typename T> struct [[clang::preferred_name(string),
                              clang::preferred_name(wstring)]] basic_string {
  // ...
};
```

Note that the `preferred_name` attribute will be ignored when the compiler
writes a C++20 Module interface now. This is due to a compiler issue
(<https://github.com/llvm/llvm-project/issues/56490>) that blocks users to modularize
declarations with `preferred_name`. This is intended to be fixed in the future.


### randomize_layout, no_randomize_layout

{clang-attr-syntaxes}`ClangRandomizeLayoutDocs`

The attribute `randomize_layout`, when attached to a C structure, selects it
for structure layout field randomization; a compile-time hardening technique. A
"seed" value, is specified via the `-frandomize-layout-seed=` command line flag.
For example:

```bash
SEED=`od -A n -t x8 -N 32 /dev/urandom | tr -d ' \n'`
make ... CFLAGS="-frandomize-layout-seed=$SEED" ...
```

You can also supply the seed in a file with `-frandomize-layout-seed-file=`.
For example:

```bash
od -A n -t x8 -N 32 /dev/urandom | tr -d ' \n' > /tmp/seed_file.txt
make ... CFLAGS="-frandomize-layout-seed-file=/tmp/seed_file.txt" ...
```

The randomization is deterministic based for a given seed, so the entire
program should be compiled with the same seed, but keep the seed safe
otherwise.

The attribute `no_randomize_layout`, when attached to a C structure,
instructs the compiler that this structure should not have its field layout
randomized.


### selectany

{clang-attr-syntaxes}`SelectAnyDocs`

This attribute appertains to a global symbol, causing it to have a weak
definition ([linkonce](https://llvm.org/docs/LangRef.html#linkage-types)),
allowing the linker to select any definition.

For more information see
[gcc documentation](https://gcc.gnu.org/onlinedocs/gcc-7.2.0/gcc/Microsoft-Windows-Variable-Attributes.html)
or [msvc documentation](https://docs.microsoft.com/pl-pl/cpp/cpp/selectany).


### transparent_union

{clang-attr-syntaxes}`TransparentUnionDocs`

This attribute can be applied to a union to change the behavior of calls to
functions that have an argument with a transparent union type. The compiler
behavior is changed in the following manner:

- A value whose type is any member of the transparent union can be passed as an
  argument without the need to cast that value.
- The argument is passed to the function using the calling convention of the
  first member of the transparent union. Consequently, all the members of the
  transparent union should have the same calling convention as its first member.

Transparent unions are not supported in C++.


### trivial_abi

{clang-attr-syntaxes}`TrivialABIDocs`

The `trivial_abi` attribute can be applied to a C++ class, struct, or union.
It instructs the compiler to pass and return the type using the C ABI for the
underlying type when the type would otherwise be considered non-trivial for the
purpose of calls.
A class annotated with `trivial_abi` can have non-trivial destructors or
copy/move constructors without automatically becoming non-trivial for the
purposes of calls. For example:

```c++
// A is trivial for the purposes of calls because `trivial_abi` makes the
// user-provided special functions trivial.
struct __attribute__((trivial_abi)) A {
  ~A();
  A(const A &);
  A(A &&);
  int x;
};

// B's destructor and copy/move constructor are considered trivial for the
// purpose of calls because A is trivial.
struct B {
  A a;
};
```

If a type is trivial for the purposes of calls, has a non-trivial destructor,
and is passed as an argument by value, the convention is that the callee will
destroy the object before returning. The lifetime of the copy of the parameter
in the caller ends without a destructor call when the call begins.

If a type is trivial for the purpose of calls, it is assumed to be trivially
relocatable for the purpose of `__is_trivially_relocatable` and
`__builtin_is_cpp_trivially_relocatable`.
When a type marked with `[[trivial_abi]]` is used as a function argument,
the compiler may omit the call to the copy constructor.
Thus, side effects of the copy constructor are potentially not performed.
For example, objects that contain pointers to themselves or otherwise depend
on their address (or the address or their subobjects) should not be declared
`[[trivial_abi]]`.

Attribute `trivial_abi` has no effect in the following cases:

- The class directly declares a virtual base or virtual methods.

- Copy constructors and move constructors of the class are all deleted.

- The class has a base class that is non-trivial for the purposes of calls.

- The class has a non-static data member whose type is non-trivial for the
  purposes of calls, which includes:

  - classes that are non-trivial for the purposes of calls
  - `__weak`-qualified types in Objective-C++
  - arrays of any of the above


### using_if_exists

{clang-attr-syntaxes}`UsingIfExistsDocs`

The `using_if_exists` attribute applies to a using-declaration. It allows
programmers to import a declaration that potentially does not exist, instead
deferring any errors to the point of use. For instance:

```c++
namespace empty_namespace {};
__attribute__((using_if_exists))
using empty_namespace::does_not_exist; // no error!

does_not_exist x; // error: use of unresolved 'using_if_exists'
```

The C++ spelling of the attribute (`[[clang::using_if_exists]]`) is also
supported as a clang extension, since ISO C++ doesn't support attributes in this
position. If the entity referred to by the using-declaration is found by name
lookup, the attribute has no effect. This attribute is useful for libraries
(primarily, libc++) that wish to redeclare a set of declarations in another
namespace, when the availability of those declarations is difficult or
impossible to detect at compile time with the preprocessor.


### weak

{clang-attr-syntaxes}`WeakDocs`

In supported output formats the `weak` attribute can be used to
specify that a variable or function should be emitted as a symbol with
`weak` (if a definition) or `extern_weak` (if a declaration of an
external symbol) [linkage](https://llvm.org/docs/LangRef.html#linkage-types).

If there is a non-weak definition of the symbol the linker will select
that over the weak. They must have same type and alignment (variables
must also have the same size), but may have a different value.

If there are multiple weak definitions of same symbol, but no non-weak
definition, they should have same type, size, alignment and value, the
linker will select one of them (see also [selectany] attribute).

If the `weak` attribute is applied to a `const` qualified variable
definition that variable is no longer consider a compiletime constant
as its value can change during linking (or dynamic linking). This
means that it can e.g no longer be part of an initializer expression.

```c
const int ANSWER __attribute__ ((weak)) = 42;

/* This function may be replaced link-time */
__attribute__ ((weak)) void debug_log(const char *msg)
{
    fprintf(stderr, "DEBUG: %s\n", msg);
}

int main(int argc, const char **argv)
{
    debug_log ("Starting up...");

    /* This may print something else than "6 * 7 = 42",
       if there is a non-weak definition of "ANSWER" in
       an object linked in */
    printf("6 * 7 = %d\n", ANSWER);

    return 0;
 }
```

If an external declaration is marked weak and that symbol does not
exist during linking (possibly dynamic) the address of the symbol will
evaluate to NULL.

```c
void may_not_exist(void) __attribute__ ((weak));

int main(int argc, const char **argv)
{
    if (may_not_exist) {
        may_not_exist();
    } else {
        printf("Function did not exist\n");
    }
    return 0;
}
```


