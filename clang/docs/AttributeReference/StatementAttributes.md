## Statement Attributes



### #pragma clang loop

{clang-attr-syntaxes}`LoopHintDocs`

The `#pragma clang loop` directive allows loop optimization hints to be
specified for the subsequent loop. The directive allows pipelining to be
disabled, or vectorization, vector predication, interleaving, and unrolling to
be enabled or disabled. Vector width, vector predication, interleave count,
unrolling count, and the initiation interval for pipelining can be explicitly
specified. See
{ref}`loop hint optimizations <langext-loop-hint-optimizations>` for details.


### #pragma clang loop pipeline, #pragma clang loop pipeline_initiation_interval

{clang-attr-syntaxes}`PipelineHintDocs`

Software Pipelining optimization is a technique used to optimize loops by
utilizing instruction-level parallelism. It reorders loop instructions to
overlap iterations. As a result, the next iteration starts before the previous
iteration has finished. The module scheduling technique creates a schedule for
one iteration such that when repeating at regular intervals, no inter-iteration
dependencies are violated. This constant interval(in cycles) between the start
of iterations is called the initiation interval. i.e. The initiation interval
is the number of cycles between two iterations of an unoptimized loop in the
newly created schedule. A new, optimized loop is created such that a single iteration
of the loop executes in the same number of cycles as the initiation interval.

For further details see <https://llvm.org/pubs/2005-06-17-LattnerMSThesis-book.pdf>.

`#pragma clang loop pipeline and #pragma loop pipeline_initiation_interval`
could be used as hints for the software pipelining optimization. The pragma is
placed immediately before a for, while, do-while, or a C++11 range-based for
loop.

Using `#pragma clang loop pipeline(disable)` avoids the software pipelining
optimization. The disable state can only be specified:

```c++
#pragma clang loop pipeline(disable)
for (...) {
  ...
}
```

Using `#pragma loop pipeline_initiation_interval` instructs
the software pipeliner to try the specified initiation interval.
If a schedule was found then the resulting loop iteration would have
the specified cycle count. If a schedule was not found then loop
remains unchanged. The initiation interval must be a positive number
greater than zero:

```c++
#pragma loop pipeline_initiation_interval(10)
for (...) {
  ...
}
```


### #pragma unroll, #pragma nounroll

{clang-attr-syntaxes}`UnrollHintDocs`

Loop unrolling optimization hints can be specified with `#pragma unroll` and
`#pragma nounroll`. The pragma is placed immediately before a for, while,
do-while, or c++11 range-based for loop. GCC's loop unrolling hints
`#pragma GCC unroll` and `#pragma GCC nounroll` are also supported and have
identical semantics to `#pragma unroll` and `#pragma nounroll`.

Specifying `#pragma unroll` without a parameter directs the loop unroller to
attempt to fully unroll the loop if the trip count is known at compile time and
attempt to partially unroll the loop if the trip count is not known at compile
time:

```c++
#pragma unroll
for (...) {
  ...
}
```

Specifying the optional parameter, `#pragma unroll _value_`, directs the
unroller to unroll the loop `_value_` times. The parameter may optionally be
enclosed in parentheses:

```c++
#pragma unroll 16
for (...) {
  ...
}

#pragma unroll(16)
for (...) {
  ...
}
```

Specifying `#pragma nounroll` indicates that the loop should not be unrolled:

```c++
#pragma nounroll
for (...) {
  ...
}
```

`#pragma unroll` and `#pragma unroll _value_` have identical semantics to
`#pragma clang loop unroll(enable)` and
`#pragma clang loop unroll_count(_value_)` respectively. `#pragma nounroll`
is equivalent to `#pragma clang loop unroll(disable)`. See
{ref}`loop hint optimizations <langext-loop-hint-optimizations>` for further
details including limitations of the unroll hints.


### [loop]

{clang-attr-syntaxes}`HLSLLoopHintDocs`

The `[loop]` directive allows loop optimization hints to be
specified for the subsequent loop. The directive allows unrolling to
be disabled and is not compatible with `[unroll(x)]`.

Specifying the parameter, `[loop]`, directs the
unroller to not unroll the loop.

```hlsl
[loop]
for (...) {
  ...
}
```

```hlsl
[loop]
while (...) {
  ...
}
```

```hlsl
[loop]
do {
  ...
} while (...)
```

See [hlsl loop extensions](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-for)
for details.


### [unroll(x)], [unroll]

{clang-attr-syntaxes}`HLSLUnrollHintDocs`

Loop unrolling optimization hints can be specified with `[unroll(x)]`
. The attribute is placed immediately before a for, while,
or do-while.
Specifying the parameter, `[unroll(_value_)]`, directs the
unroller to unroll the loop `_value_` times. Note: `[unroll(x)]` is not compatible with `[loop]`.

```hlsl
[unroll(4)]
for (...) {
  ...
}
```

```hlsl
[unroll]
for (...) {
  ...
}
```

```hlsl
[unroll(4)]
while (...) {
  ...
}
```

```hlsl
[unroll]
while (...) {
  ...
}
```

```hlsl
[unroll(4)]
do {
  ...
} while (...)
```

```hlsl
[unroll]
do {
  ...
} while (...)
```

See [hlsl loop extensions](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-for)
for details.


### __read_only, __write_only, __read_write (read_only, write_only, read_write)

{clang-attr-syntaxes}`OpenCLAccessDocs`

The access qualifiers must be used with image object arguments or pipe arguments
to declare if they are being read or written by a kernel or function.

The `read_only`, `__read_only`, `write_only`, `__write_only`, `read_write`, and `__read_write`
names are reserved for use as access qualifiers and shall not be used otherwise.

```c
kernel void
foo (read_only image2d_t imageA,
     write_only image2d_t imageB) {
  ...
}
```

In the above example imageA is a read-only 2D image object, and imageB is a
write-only 2D image object.

The `read_write` (or `__read_write`) qualifier cannot be used with pipe arguments.

More details can be found in the OpenCL C language Spec v2.0, Section 6.6.


### assume

{clang-attr-syntaxes}`CXXAssumeDocs`

The `assume` attribute is used to indicate to the optimizer that a
certain condition is assumed to be true at a certain point in the
program. If this condition is violated at runtime, the behavior is
undefined. `assume` can only be applied to a null statement.

Different optimisers are likely to react differently to the presence of
this attribute; in some cases, adding `assume` may affect performance
negatively. It should be used with parsimony and care.

Example:

```c++
int f(int x, int y) {
  [[assume(x == 27)]];
  [[assume(x == y)]];
  return y + 1; // May be optimised to `return 28`.
}
```


### atomic

{clang-attr-syntaxes}`AtomicDocs`

The `atomic` attribute can be applied to *compound statements* to override or
further specify the default atomic code-generation behavior, especially on
targets such as AMDGPU. You can annotate compound statements with options
to modify how atomic instructions inside that statement are emitted at the IR
level.

For details, see the documentation for
{ref}`@atomic <langext-atomic-code-generation>`


### branch, flatten

{clang-attr-syntaxes}`HLSLControlFlowHintDocs`

The `branch` and `flatten` attributes can be applied to *if* and *switch*
statements in the HLSL language mode to provide hints for how the backend
should execute them.

- `branch` means that control flow is preferred. The condition should be
  evaluated first and we should only execute the block guarded by it.

- `flatten` means that control flow should be avoided. All blocks should be
  executed and variables that are modified should be conditionally assigned.

These control flow hints are preserved through the compilation and emitted in a
backend-specific way.

For details, see the Direct3D documentation for [if Statement][if Statement]
and [switch Statement][switch Statement].

[if Statement]: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-if
[switch Statement]: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-switch


### constexpr

{clang-attr-syntaxes}`MSConstexprDocs`

The `[[msvc::constexpr]]` attribute can be applied only to a function
definition or a `return` statement. It does not impact function declarations.
A `[[msvc::constexpr]]` function cannot be `constexpr` or `consteval`.
A `[[msvc::constexpr]]` function is treated as if it were a `constexpr` function
when it is evaluated in a constant context of `[[msvc::constexpr]] return` statement.
Otherwise, it is treated as a regular function.

Semantics of this attribute are enabled only under MSVC compatibility
(`-fms-compatibility-version`) 19.33 and later.


### fallthrough

{clang-attr-syntaxes}`FallthroughDocs`

The `fallthrough` (or `clang::fallthrough`) attribute is used
to annotate intentional fall-through
between switch labels. It can only be applied to a null statement placed at a
point of execution between any statement and the next switch label. It is
common to mark these places with a specific comment, but this attribute is
meant to replace comments with a more strict annotation, which can be checked
by the compiler. This attribute doesn't change semantics of the code and can
be used wherever an intended fall-through occurs. It is designed to mimic
control-flow statements like `break;`, so it can be placed in most places
where `break;` can, but only if there are no statements on the execution path
between it and the next switch label.

By default, Clang does not warn on unannotated fallthrough from one `switch`
case to another. Diagnostics on fallthrough without a corresponding annotation
can be enabled with the `-Wimplicit-fallthrough` argument.

Here is an example:

```c++
// compile with -Wimplicit-fallthrough
switch (n) {
case 22:
case 33:  // no warning: no statements between case labels
  f();
case 44:  // warning: unannotated fall-through
  g();
  [[clang::fallthrough]];
case 55:  // no warning
  if (x) {
    h();
    break;
  }
  else {
    i();
    [[clang::fallthrough]];
  }
case 66:  // no warning
  p();
  [[clang::fallthrough]]; // warning: fallthrough annotation does not
                          //          directly precede case label
  q();
case 77:  // warning: unannotated fall-through
  r();
}
```


### intel_reqd_sub_group_size

{clang-attr-syntaxes}`OpenCLIntelReqdSubGroupSizeDocs`

The optional attribute intel_reqd_sub_group_size can be used to indicate that
the kernel must be compiled and executed with the specified subgroup size. When
this attribute is present, get_max_sub_group_size() is guaranteed to return the
specified integer value. This is important for the correctness of many subgroup
algorithms, and in some cases may be used by the compiler to generate more optimal
code. See
[`cl_intel_required_subgroup_size`](https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_required_subgroup_size.html)
for details.


### likely and unlikely

{clang-attr-syntaxes}`LikelihoodDocs`

The `likely` and `unlikely` attributes are used as compiler hints.
The attributes are used to aid the compiler to determine which branch is
likely or unlikely to be taken. This is done by marking the branch substatement
with one of the two attributes.

It isn't allowed to annotate a single statement with both `likely` and
`unlikely`. Annotating the `true` and `false` branch of an `if`
statement with the same likelihood attribute will result in a diagnostic and
the attributes are ignored on both branches.

In a `switch` statement it's allowed to annotate multiple `case` labels
or the `default` label with the same likelihood attribute. This makes
\* all labels without an attribute have a neutral likelihood,
\* all labels marked `[[likely]]` have an equally positive likelihood, and
\* all labels marked `[[unlikely]]` have an equally negative likelihood.
The neutral likelihood is the more likely of path execution than the negative
likelihood. The positive likelihood is the more likely of path of execution
than the neutral likelihood.

These attributes have no effect on the generated code when using
PGO (Profile-Guided Optimization) or at optimization level 0.

In Clang, the attributes will be ignored if they're not placed on
\* the `case` or `default` label of a `switch` statement,
\* or on the substatement of an `if` or `else` statement,
\* or on the substatement of an `for` or `while` statement.
The C++ Standard recommends to honor them on every statement in the
path of execution, but that can be confusing:

```c++
if (b) {
  [[unlikely]] --b; // Per the standard this is in the path of
                    // execution, so this branch should be considered
                    // unlikely. However, Clang ignores the attribute
                    // here since it is not on the substatement.
}

if (b) {
  --b;
  if(b)
    return;
  [[unlikely]] --b; // Not in the path of execution,
}                   // the branch has no likelihood information.

if (b) {
  --b;
  foo(b);
  // Whether or not the next statement is in the path of execution depends
  // on the declaration of foo():
  // In the path of execution: void foo(int);
  // Not in the path of execution: [[noreturn]] void foo(int);
  // This means the likelihood of the branch depends on the declaration
  // of foo().
  [[unlikely]] --b;
}
```

Below are some example usages of the likelihood attributes and their effects:

```c++
if (b) [[likely]] { // Placement on the first statement in the branch.
  // The compiler will optimize to execute the code here.
} else {
}

if (b)
  [[unlikely]] b++; // Placement on the first statement in the branch.
else {
  // The compiler will optimize to execute the code here.
}

if (b) {
  [[unlikely]] b++; // Placement on the second statement in the branch.
}                   // The attribute will be ignored.

if (b) [[likely]] {
  [[unlikely]] b++; // No contradiction since the second attribute
}                   // is ignored.

if (b)
  ;
else [[likely]] {
  // The compiler will optimize to execute the code here.
}

if (b)
  ;
else
  // The compiler will optimize to execute the next statement.
  [[likely]] b = f();

if (b) [[likely]]; // Both branches are likely. A diagnostic is issued
else [[likely]];   // and the attributes are ignored.

if (b)
  [[likely]] int i = 5; // Issues a diagnostic since the attribute
                        // isn't allowed on a declaration.

switch (i) {
  [[likely]] case 1:    // This value is likely
    ...
    break;

  [[unlikely]] case 2:  // This value is unlikely
    ...
    [[fallthrough]];

  case 3:               // No likelihood attribute
    ...
    [[likely]] break;   // No effect

  case 4: [[likely]] {  // attribute on substatement has no effect
    ...
    break;
    }

  [[unlikely]] default: // All other values are unlikely
    ...
    break;
}

switch (i) {
  [[likely]] case 0:    // This value and code path is likely
    ...
    [[fallthrough]];

  case 1:               // No likelihood attribute, code path is neutral
    break;              // falling through has no effect on the likelihood

  case 2:               // No likelihood attribute, code path is neutral
    [[fallthrough]];

  [[unlikely]] default: // This value and code path are both unlikely
    break;
}

for(int i = 0; i != size; ++i) [[likely]] {
  ...               // The loop is the likely path of execution
}

for(const auto &E : Elements) [[likely]] {
  ...               // The loop is the likely path of execution
}

while(i != size) [[unlikely]] {
  ...               // The loop is the unlikely path of execution
}                   // The generated code will optimize to skip the loop body

while(true) [[unlikely]] {
  ...               // The attribute has no effect
}                   // Clang elides the comparison and generates an infinite
                    // loop
```


### musttail

{clang-attr-syntaxes}`MustTailDocs`

If a `return` statement is marked `musttail`, this indicates that the
compiler must generate a tail call for the program to be correct, even when
optimizations are disabled. This guarantees that the call will not cause
unbounded stack growth if it is part of a recursive cycle in the call graph.

If the callee is a virtual function that is implemented by a thunk, there is
no guarantee in general that the thunk tail-calls the implementation of the
virtual function, so such a call in a recursive cycle can still result in
unbounded stack growth.

`clang::musttail` can only be applied to a `return` statement whose value
is the result of a function call (even functions returning void must use
`return`, although no value is returned). The target function must have the
same number of arguments as the caller. The types of the return value and all
arguments must be similar according to C++ rules (differing only in cv
qualifiers or array size), including the implicit "this" argument, if any.
Any variables in scope, including all arguments to the function and the
return value must be trivially destructible. The calling convention of the
caller and callee must match, and they must not be variadic functions or have
old style K&R C function declarations.

The lifetimes of all local variables and function parameters end immediately
before the call to the function. This means that it is undefined behaviour to
pass a pointer or reference to a local variable to the called function, which
is not the case without the attribute. Clang will emit a warning in common
cases where this happens.

`clang::musttail` provides assurances that the tail call can be optimized on
all targets, not just one.


### nomerge

{clang-attr-syntaxes}`NoMergeDocs`

If a statement is marked `nomerge` and contains call expressions, those call
expressions inside the statement will not be merged during optimization. This
attribute can be used to prevent the optimizer from obscuring the source
location of certain calls. For example, it will prevent tail merging otherwise
identical code sequences that raise an exception or terminate the program. Tail
merging normally reduces the precision of source location information, making
stack traces less useful for debugging. This attribute gives the user control
over the tradeoff between code size and debug information precision.

`nomerge` attribute can also be used as function attribute to prevent all
calls to the specified function from merging. It has no effect on indirect
calls to such functions. For example:

```c++
[[clang::nomerge]] void foo(int) {}

void bar(int x) {
  auto *ptr = foo;
  if (x) foo(1); else foo(2); // will not be merged
  if (x) ptr(1); else ptr(2); // indirect call, can be merged
}
```

`nomerge` attribute can also be used for pointers to functions to
prevent calls through such pointer from merging. In such case the
effect applies only to a specific function pointer. For example:

```c++
[[clang::nomerge]] void (*foo)(int);

void bar(int x) {
  auto *ptr = foo;
  if (x) foo(1); else foo(2); // will not be merged
  if (x) ptr(1); else ptr(2); // 'ptr' has no 'nomerge' attribute, can be merged
}
```


### opencl_unroll_hint

{clang-attr-syntaxes}`OpenCLUnrollHintDocs`

The `opencl_unroll_hint` attribute qualifier can be used to specify that a loop
(for, while and do loops) can be unrolled. This attribute qualifier can be
used to specify full unrolling or partial unrolling by a specified amount.
This is a compiler hint and the compiler may ignore this directive. See
[OpenCL v2.0](https://www.khronos.org/registry/cl/specs/opencl-2.0.pdf)
s6.11.5 for details.


### suppress

{clang-attr-syntaxes}`SuppressDocs`

The `suppress` attribute suppresses unwanted warnings coming from static
analysis tools such as the Clang Static Analyzer. The tool will not report
any issues in source code annotated with the attribute.

The attribute cannot be used to suppress traditional Clang warnings, because
many such warnings are emitted before the attribute is fully parsed.
Consider using `#pragma clang diagnostic` to control such diagnostics,
as described in
{ref}`Controlling Diagnostics via Pragmas <pragma-gcc-diagnostic>`.

The `suppress` attribute can be placed on an individual statement in order to
suppress warnings about undesirable behavior occurring at that statement:

```c++
int foo() {
  int *x = nullptr;
  ...
  [[clang::suppress]]
  return *x;  // null pointer dereference warning suppressed here
}
```

Putting the attribute on a compound statement suppresses all warnings in scope:

```c++
int foo() {
  [[clang::suppress]] {
    int *x = nullptr;
    ...
    return *x;  // warnings suppressed in the entire scope
  }
}
```

The attribute can also be placed on entire declarations of functions, classes,
variables, member variables, and so on, to suppress warnings related
to the declarations themselves. When used this way, the attribute additionally
suppresses all warnings in the lexical scope of the declaration:

```c++
class [[clang::suppress]] C {
  int foo() {
    int *x = nullptr;
    ...
    return *x;  // warnings suppressed in the entire class scope
  }

  int bar();
};

int C::bar() {
  int *x = nullptr;
  ...
  return *x;  // warning NOT suppressed! - not lexically nested in 'class C{}'
}
```

Some static analysis warnings are accompanied by one or more notes, and the
line of code against which the warning is emitted isn't necessarily the best
for suppression purposes. In such cases the tools are allowed to implement
additional ways to suppress specific warnings based on the attribute attached
to a note location.

For example, the Clang Static Analyzer suppresses memory leak warnings when
the suppression attribute is placed at the allocation site (highlited by
a "note: memory is allocated"), which may be different from the line of code
at which the program "loses track" of the pointer (where the warning
is ultimately emitted):

```c
int bar1(bool coin_flip) {
  __attribute__((suppress))
  int *result = (int *)malloc(sizeof(int));
  if (coin_flip)
    return 1;  // warning about this leak path is suppressed

  return *result;  // warning about this leak path is also suppressed
}

int bar2(bool coin_flip) {
  int *result = (int *)malloc(sizeof(int));
  if (coin_flip)
    return 1;  // leak warning on this path NOT suppressed

  __attribute__((suppress))
  return *result;  // leak warning is suppressed only on this path
}
```

When written as `[[gsl::suppress]]`, this attribute suppresses specific
clang-tidy diagnostics for rules of the [C++ Core Guidelines][c++ core guidelines] in a portable
way. The attribute can be attached to declarations, statements, and at
namespace scope.

```c++
[[gsl::suppress("Rh-public")]]
void f_() {
  int *p;
  [[gsl::suppress("type")]] {
    p = reinterpret_cast<int*>(7);
  }
}
namespace N {
  [[clang::suppress("type", "bounds")]];
  ...
}
```

[c++ core guidelines]: https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#inforce-enforcement


### sycl_special_class

{clang-attr-syntaxes}`SYCLSpecialClassDocs`

SYCL defines some special classes (accessor, sampler, and stream) which require
specific handling during the generation of the SPIR entry point.
The `__attribute__((sycl_special_class))` attribute is used in SYCL
headers to indicate that a class or a struct needs a specific handling when
it is passed from host to device.
Special classes will have a mandatory `__init` method and an optional
`__finalize` method (the `__finalize` method is used only with the
`stream` type). Kernel parameters types are extract from the `__init` method
parameters. The kernel function arguments list is derived from the
arguments of the `__init` method. The arguments of the `__init` method are
copied into the kernel function argument list and the `__init` and
`__finalize` methods are called at the beginning and the end of the kernel,
respectively.
The `__init` and `__finalize` methods must be defined inside the
special class.
Please note that this is an attribute that is used as an internal
implementation detail and not intended to be used by external users.

The syntax of the attribute is as follows:

```text
class __attribute__((sycl_special_class)) accessor {};
class [[clang::sycl_special_class]] accessor {};
```

This is a code example that illustrates the use of the attribute:

```c++
class __attribute__((sycl_special_class)) SpecialType {
  int F1;
  int F2;
  void __init(int f1) {
    F1 = f1;
    F2 = f1;
  }
  void __finalize() {}
public:
  SpecialType() = default;
  int getF2() const { return F2; }
};

int main () {
  SpecialType T;
  cgh.single_task([=] {
    T.getF2();
  });
}
```

This would trigger the following kernel entry point in the AST:

```c++
void __sycl_kernel(int f1) {
  SpecialType T;
  T.__init(f1);
  ...
  T.__finalize()
}
```


