# SanitizerCoverage


## Introduction

LLVM has a simple code coverage instrumentation built in (SanitizerCoverage).
It inserts calls to user-defined functions on function-, basic-block-, and edge- levels.
Default implementations of those callbacks are provided and implement
simple coverage reporting and visualization,
however if you need *just* coverage visualization you may want to use
{doc}`SourceBasedCodeCoverage <SourceBasedCodeCoverage>` instead.

## Tracing PCs with guards

With `-fsanitize-coverage=trace-pc-guard` the compiler will insert the following code
on every edge:

```none
__sanitizer_cov_trace_pc_guard(&guard_variable)
```

Every edge will have its own `guard_variable` (uint32_t).

The compiler will also insert calls to a module constructor:

```c++
// The guards are [start, stop).
// This function will be called at least once per DSO and may be called
// more than once with the same values of start/stop.
__sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop);
```

With an additional `...=trace-pc,indirect-calls` flag
`__sanitizer_cov_trace_pc_indirect(void *callee)` will be inserted on every indirect call.

The functions `__sanitizer_cov_trace_pc_*` should be defined by the user.

Example:

```c++
// trace-pc-guard-cb.cc
#include <stdint.h>
#include <stdio.h>
#include <sanitizer/coverage_interface.h>

// This callback is inserted by the compiler as a module constructor
// into every DSO. 'start' and 'stop' correspond to the
// beginning and end of the section with the guards for the entire
// binary (executable or DSO). The callback will be called at least
// once per DSO and may be called multiple times with the same parameters.
extern "C" void __sanitizer_cov_trace_pc_guard_init(uint32_t *start,
                                                    uint32_t *stop) {
  static uint64_t N;  // Counter for the guards.
  if (start == stop || *start) return;  // Initialize only once.
  printf("INIT: %p %p\n", start, stop);
  for (uint32_t *x = start; x < stop; x++)
    *x = ++N;  // Guards should start from 1.
}

// This callback is inserted by the compiler on every edge in the
// control flow (some optimizations apply).
// Typically, the compiler will emit the code like this:
//    if(*guard)
//      __sanitizer_cov_trace_pc_guard(guard);
// But for large functions it will emit a simple call:
//    __sanitizer_cov_trace_pc_guard(guard);
extern "C" void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
  if (!*guard) return;  // Duplicate the guard check.
  // If you set *guard to 0 this code will not be called again for this edge.
  // Now you can get the PC and do whatever you want:
  //   store it somewhere or symbolize it and print right away.
  // The values of `*guard` are as you set them in
  // __sanitizer_cov_trace_pc_guard_init and so you can make them consecutive
  // and use them to dereference an array or a bit vector.
  void *PC = __builtin_return_address(0);
  char PcDescr[1024];
  // This function is a part of the sanitizer run-time.
  // To use it, link with AddressSanitizer or other sanitizer.
  __sanitizer_symbolize_pc(PC, "%p %F %L", PcDescr, sizeof(PcDescr));
  printf("guard: %p %x PC %s\n", guard, *guard, PcDescr);
}
```

```c++
// trace-pc-guard-example.cc
void foo() { }
int main(int argc, char **argv) {
  if (argc > 1) foo();
}
```

```console
clang++ -g  -fsanitize-coverage=trace-pc-guard trace-pc-guard-example.cc -c
clang++ trace-pc-guard-cb.cc trace-pc-guard-example.o -fsanitize=address
ASAN_OPTIONS=strip_path_prefix=`pwd`/ ./a.out
```

```console
INIT: 0x71bcd0 0x71bce0
guard: 0x71bcd4 2 PC 0x4ecd5b in main trace-pc-guard-example.cc:2
guard: 0x71bcd8 3 PC 0x4ecd9e in main trace-pc-guard-example.cc:3:7
```

```console
ASAN_OPTIONS=strip_path_prefix=`pwd`/ ./a.out with-foo
```

```console
INIT: 0x71bcd0 0x71bce0
guard: 0x71bcd4 2 PC 0x4ecd5b in main trace-pc-guard-example.cc:3
guard: 0x71bcdc 4 PC 0x4ecdc7 in main trace-pc-guard-example.cc:4:17
guard: 0x71bcd0 1 PC 0x4ecd20 in foo() trace-pc-guard-example.cc:2:14
```

## Inline 8bit-counters

**Experimental, may change or disappear in future**

With `-fsanitize-coverage=inline-8bit-counters` the compiler will insert
inline counter increments on every edge.
This is similar to `-fsanitize-coverage=trace-pc-guard` but instead of a
callback the instrumentation simply increments a counter.

Users need to implement a single function to capture the counters at startup.

```c++
extern "C"
void __sanitizer_cov_8bit_counters_init(char *start, char *end) {
  // [start,end) is the array of 8-bit counters created for the current DSO.
  // Capture this array in order to read/modify the counters.
}
```

## Inline bool-flag

**Experimental, may change or disappear in future**

With `-fsanitize-coverage=inline-bool-flag` the compiler will insert
setting an inline boolean to true on every edge.
This is similar to `-fsanitize-coverage=inline-8bit-counter` but instead of
an increment of a counter, it just sets a boolean to true.

Users need to implement a single function to capture the flags at startup.

```c++
extern "C"
void __sanitizer_cov_bool_flag_init(bool *start, bool *end) {
  // [start,end) is the array of boolean flags created for the current DSO.
  // Capture this array in order to read/modify the flags.
}
```

## PC-Table

**Experimental, may change or disappear in future**

**Note:** this instrumentation might be incompatible with dead code stripping
(`-Wl,-gc-sections`) for linkers other than LLD, thus resulting in a
significant binary size overhead. For more information, see
[Bug 34636](https://bugs.llvm.org/show_bug.cgi?id=34636).

With `-fsanitize-coverage=pc-table` the compiler will create a table of
instrumented PCs. Requires either `-fsanitize-coverage=inline-8bit-counters`,
or `-fsanitize-coverage=inline-bool-flag`, or `-fsanitize-coverage=trace-pc-guard`.

Users need to implement a single function to capture the PC table at startup:

```c++
extern "C"
void __sanitizer_cov_pcs_init(const uintptr_t *pcs_beg,
                              const uintptr_t *pcs_end) {
  // [pcs_beg,pcs_end) is the array of ptr-sized integers representing
  // pairs [PC,PCFlags] for every instrumented block in the current DSO.
  // Capture this array in order to read the PCs and their Flags.
  // The number of PCs and PCFlags for a given DSO is the same as the number
  // of 8-bit counters (-fsanitize-coverage=inline-8bit-counters), or
  // boolean flags (-fsanitize-coverage=inline=bool-flags), or trace_pc_guard
  // callbacks (-fsanitize-coverage=trace-pc-guard).
  // A PCFlags describes the basic block:
  //  * bit0: 1 if the block is the function entry block, 0 otherwise.
}
```

## Tracing PCs

With `-fsanitize-coverage=trace-pc` the compiler will insert
`__sanitizer_cov_trace_pc()` on every edge.
With an additional `...=trace-pc,indirect-calls` flag
`__sanitizer_cov_trace_pc_indirect(void *callee)` will be inserted on every indirect call.

With `-fsanitize-coverage=trace-pc-entry-exit` the compiler will insert
`__sanitizer_cov_trace_pc_entry()` on function entry, and insert
`__sanitizer_cov_trace_pc_exit()` on function return;
`-fsanitize-coverage=trace-pc` or `-fsanitize-coverage=trace-pc-guard` must
still be passed to instrument all basic blocks.

With the combination `-fsanitize-coverage=trace-pc-entry-exit,trace-pc`,
`__sanitizer_cov_trace_pc()` will be omitted in the entry basic block because
the block is already covered by `__sanitizer_cov_trace_pc_entry()`, which can
be used to both record that the function has been entered and record coverage of
the entry basic block.
However, with `-fsanitize-coverage=trace-pc-entry-exit,trace-pc-guard`, both
callbacks are called for the entry block because
`__sanitizer_cov_trace_pc_entry()` does not provide a `guard_variable`.

These callbacks are not implemented in the Sanitizer run-time and should be defined
by the user.
This mechanism is used for fuzzing the Linux kernel
(<https://github.com/google/syzkaller>).

## Instrumentation points

Sanitizer Coverage offers different levels of instrumentation.

- `edge` (default): edges are instrumented (see below).
- `bb`: basic blocks are instrumented.
- `func`: only the entry block of every function will be instrumented.

Use these flags together with `trace-pc-guard` or `trace-pc`,
like this: `-fsanitize-coverage=func,trace-pc-guard`.

When `edge` or `bb` is used, some of the edges/blocks may still be left
uninstrumented (pruned) if such instrumentation is considered redundant.
Use `no-prune` (e.g. `-fsanitize-coverage=bb,no-prune,trace-pc-guard`)
to disable pruning. This could be useful for better coverage visualization.

### Edge coverage

Consider this code:

```c++
void foo(int *a) {
  if (a)
    *a = 0;
}
```

It contains 3 basic blocks, let's name them A, B, C:

```none
A
|\
| \
|  B
| /
|/
C
```

If blocks A, B, and C are all covered we know for certain that the edges A=>B
and B=>C were executed, but we still don't know if the edge A=>C was executed.
Such edges of control flow graph are called
[critical](https://en.wikipedia.org/wiki/Control_flow_graph#Special_edges).
The edge-level coverage simply splits all critical edges by introducing new
dummy blocks and then instruments those blocks:

```none
A
|\
| \
D  B
| /
|/
C
```

## Tracing data flow

Support for data-flow-guided fuzzing.
With `-fsanitize-coverage=trace-cmp` the compiler will insert extra instrumentation
around comparison instructions and switch statements.
Similarly, with `-fsanitize-coverage=trace-div` the compiler will instrument
integer division instructions (to capture the right argument of division)
and with `-fsanitize-coverage=trace-gep` --
the [LLVM GEP instructions](https://llvm.org/docs/GetElementPtr.html)
(to capture array indices).
Similarly, with `-fsanitize-coverage=trace-loads` and `-fsanitize-coverage=trace-stores`
the compiler will instrument loads and stores, respectively.

Currently, these flags do not work by themselves - they require one
of `-fsanitize-coverage={trace-pc,inline-8bit-counters,inline-bool}`
flags to work.

Unless `no-prune` option is provided, some of the comparison instructions
will not be instrumented.

```c++
// Called before a comparison instruction.
// Arg1 and Arg2 are arguments of the comparison.
void __sanitizer_cov_trace_cmp1(uint8_t Arg1, uint8_t Arg2);
void __sanitizer_cov_trace_cmp2(uint16_t Arg1, uint16_t Arg2);
void __sanitizer_cov_trace_cmp4(uint32_t Arg1, uint32_t Arg2);
void __sanitizer_cov_trace_cmp8(uint64_t Arg1, uint64_t Arg2);

// Called before a comparison instruction if exactly one of the arguments is constant.
// Arg1 and Arg2 are arguments of the comparison, Arg1 is a compile-time constant.
// These callbacks are emitted by -fsanitize-coverage=trace-cmp since 2017-08-11
void __sanitizer_cov_trace_const_cmp1(uint8_t Arg1, uint8_t Arg2);
void __sanitizer_cov_trace_const_cmp2(uint16_t Arg1, uint16_t Arg2);
void __sanitizer_cov_trace_const_cmp4(uint32_t Arg1, uint32_t Arg2);
void __sanitizer_cov_trace_const_cmp8(uint64_t Arg1, uint64_t Arg2);

// Called before a switch statement.
// Val is the switch operand.
// Cases[0] is the number of case constants.
// Cases[1] is the size of Val in bits.
// Cases[2:] are the case constants.
void __sanitizer_cov_trace_switch(uint64_t Val, uint64_t *Cases);

// Called before a division statement.
// Val is the second argument of division.
void __sanitizer_cov_trace_div4(uint32_t Val);
void __sanitizer_cov_trace_div8(uint64_t Val);

// Called before a GetElementPtr (GEP) instruction
// for every non-constant array index.
void __sanitizer_cov_trace_gep(uintptr_t Idx);

// Called before a load of appropriate size. Addr is the address of the load.
void __sanitizer_cov_load1(uint8_t *addr);
void __sanitizer_cov_load2(uint16_t *addr);
void __sanitizer_cov_load4(uint32_t *addr);
void __sanitizer_cov_load8(uint64_t *addr);
void __sanitizer_cov_load16(__int128 *addr);
// Called before a store of appropriate size. Addr is the address of the store.
void __sanitizer_cov_store1(uint8_t *addr);
void __sanitizer_cov_store2(uint16_t *addr);
void __sanitizer_cov_store4(uint32_t *addr);
void __sanitizer_cov_store8(uint64_t *addr);
void __sanitizer_cov_store16(__int128 *addr);
```

## Tracing function arguments and return values

With `-fsanitize-coverage=trace-args` the compiler will insert a callback for
each argument of every instrumented function, on entry, and with
`-fsanitize-coverage=trace-ret` a callback before every return. Neither flag
needs another coverage mode: used alone, they emit no block or edge callbacks.

The design rule for both flags is:

> Debug information may enrich the interpretation of an observation, but must
> never determine which instrumentation is emitted.

What is observed is decided by the LLVM IR of the instrumented function alone.
A build with `-g` emits exactly the same callbacks, with the same operands, and
the same machine code, as a build without it; debug information gives the
calls a source location and nothing else.

```c++
// Called once per observed IR argument, or element of one, on entry.
// tag:        (arg_idx << 32) | size, where arg_idx is the zero-based
//             position in the IR parameter list and size is the number of
//             bytes reported (0 if nothing to report)
// val:        the value itself, or the address of the bytes to read
// offsets:    null if `val` is a value; otherwise a constant table:
//             offsets[0] is the number of pairs N, followed by N
//             [byte offset, byte size] pairs, none wider than 8 bytes
void __sanitizer_cov_trace_args(uint64_t tag, uint64_t val,
                                const uint64_t *offsets);

// Called before each return; once per element of a first-class aggregate
// return value.
// size:       number of bytes reported, 0 if nothing to report
void __sanitizer_cov_trace_ret(uint64_t size, uint64_t val,
                               const uint64_t *offsets);
```

Neither callback is told which function it speaks for. A runtime that needs to
know takes its own return address, less one so that the address lies within the
call rather than on the instruction after it, and resolves it like any other
code address. This is the same convention as the `trace-cmp` callbacks. Every
observation has a call of its own, which is never a tail call, so that address
identifies it.

`offsets` says what `val` holds:

* `offsets == nullptr`: `val` **is** the value, its low `size` bytes,
  zero-extended. Integers, pointers, floating-point values and vectors are
  reported this way: a pointer as the address it holds, a floating-point value
  or a vector as its bit pattern, and a value wider than 64 bits by its low 64
  bits.

* `offsets != nullptr`: `val` is the **address** of `size` bytes that may be
  read at the call, and `offsets` says which: `offsets[0]` pieces follow it as
  [byte offset, byte size] pairs, none wider than 8 bytes, so a consumer reading
  each into a 64-bit word reads all of them. That is one of

  * an object the ABI placed in memory, which the IR itself marks as one: an
    argument with a `byval`, `byref`, `inalloca` or `preallocated` attribute,
    or the `sret` buffer of an indirect return. `offsets` describes the
    elements of its IR type, nested structs included; anything wider than
    8 bytes - an array, or the opaque `[N x i8]` rustc types every such object
    as - in 8-byte pieces.
  * a *pointee snapshot*: a `noundef` pointer argument the IR says is
    `dereferenceable(N)` or `dereferenceable_or_null(N)` on entry, or a
    pointer return value the function's return attributes say is (`noundef`
    included), at the return. `val` is the pointer, and `size` is the first
    min(N, `-sanitizer-coverage-trace-pointee-bytes`) bytes it points to
    (64 by default; 0 reports every pointer by value), covered in 8-byte
    pieces. The IR attribute alone bounds the read - the IR has no pointee
    type - so the bytes are raw; a consumer cuts them into members with the
    binary's debug information. A
    `dereferenceable_or_null` pointer may be null, which the runtime checks.

  The callback does not say which of the two it is. Every other
  pointer is reported by value, and what it points to is never read: the
  instrumentation reads memory only as far as the IR proves it may.

`size == 0` means there was nothing to report: a `void` return, a value with no
fixed-size bit pattern, such as a scalable vector, or an object the function
does not use. Callers may pass anything for an argument the function does not
use - dead argument elimination passes poison in the callers it can see - so
such an object is not handed to the runtime to read; an unused value is still
reported. A value that may be poison is frozen first, so `val` always holds
bits.

The compiler never creates memory to report a value from, so no stack slot
escapes and no frame is realigned on the instrumentation's account.

### IR arguments, not source parameters

`arg_idx` is the position of the argument in the IR function's parameter list.
That is the ABI's view of the signature, and it is not guaranteed to equal the
source-language parameter number:

* One source parameter may be several IR arguments. On x86-64,
  `void f(struct mix { double d; long l; } m, int z)` is
  `f(double %0, i64 %1, i32 %2)`: `m` is IR arguments 0 and 1, and `z` is IR
  argument 2.
* One IR argument may be several observations. On AArch64 the same function is
  `f([2 x i64] %0, i32 %1)`: the first-class aggregate `%0` is reported as two
  callbacks, both with `arg_idx == 0`, and `z` is IR argument 1.
* A hidden struct-return (`sret`) pointer and a Swift error slot are not
  observed, but keep their positions: in `struct big f(long x)`, lowered to
  `f(ptr sret %0, i64 %1)`, `x` is reported with `arg_idx == 1`.
* A C++ `this` is an ordinary pointer argument, reported as its value - with
  the `sizeof` bytes of the object it points to, which the IR says are
  dereferenceable, as for a C++ reference.

A struct the ABI passes by address without an IR attribute that says so - a
large struct on AArch64, a non-trivially-copyable C++ object - is a pointer
argument, and is reported as the pointer, with what it points to if the IR
says it is dereferenceable.

Which pointers the IR says that of depends on the front end. Clang gives C++
references and `this` `dereferenceable(sizeof(T))`, as it does a struct the ABI
passes by address (on AArch64, any over 16 bytes), and a C array parameter of
static size, `const struct s p[static 1]`, `dereferenceable` that far; a plain
C pointer has no such attribute, and is reported by value. rustc gives `&T` and
`&mut T` arguments `dereferenceable(size_of::<T>())`, `Option<&T>`
`dereferenceable_or_null`, and the vtable of a `&dyn Trait` its size; not a
`Box<T>`, a raw pointer, the data of a slice or trait object, a `&` to a
`!Freeze` type (`Cell`, `UnsafeCell`) or a `&mut` to a `!Unpin` one, nor any
return value.

### Pointers: values, layouts and snapshots

Four things about a pointer argument or return value are easily confused, and
only the first and the third are recorded:

* The **pointer value**, the address it holds: always reported, as `val`.
* The **source-level pointee type and member layout**: what the debug
  information says it points to. A consumer can name and lay out the members
  from it, but debug information never causes anything to be read.
* An explicit **pointee snapshot**: bytes the runtime reads behind the pointer
  at the trace point. Only an explicit IR dereferenceability contract causes
  the instrumentation runtime to read behind a pointer:
  `noundef` together with `dereferenceable(N)`, or with
  `dereferenceable_or_null(N)` and a null check, and then the first
  min(N, `-sanitizer-coverage-trace-pointee-bytes`) bytes. The
  attribute gives a bounded readable range, not a type: which members those
  bytes are is the decoder's business, with the debug information.
* **Complete and partial values**: a member wholly within the bytes read is a
  complete value; one the byte limit cuts is partial, and one beyond it was
  not read.

What carries the contract is up to the front end (see above). An ordinary C
`T *` argument normally does not, and is a pointer value only; a Rust `&T` or
`&mut T` usually does; a Rust raw pointer, `*const T` or `*mut T`, does not,
nor does a C pointer unless declared `T p[static N]`. Nothing is inferred to
stand in for a missing contract - not from later accesses, the size of an
object, the callers, the debug information or what must execute - and a
pointer stored inside a snapshot is a value too: what it points to is not
read. The instrumentation therefore does not reconstruct every pointer
argument, nor complete structs behind pointers. As a measure of how far it
goes, on an x86-64 Linux kernel build the pointee of 0 of 23,810 C pointer
arguments has a byte recorded, and on arm64 21 of 22,606 (structs AArch64
passes by reference); of the 756 pointer arguments of the kernel's Rust
objects, 318 (42%) do. These are the shares of pointer arguments with at
least one recorded pointee byte, not of arguments, structs or return values
reconstructed in full.

Recording, for a pointer without a contract, the values the function itself
loads and stores through it, when it does, is possible future work, and not
part of this implementation.

Mapping an observation to a source parameter, or to a struct field, is the
job of a consumer that has the binary's debug information; the
instrumentation does not attempt it. An inlined callee is not an IR function
of its own any more, so its parameters are not observed: arguments are those of
the IR functions that remain after the inliner ran. `-fno-inline` makes more of
them visible, at the usual cost.

Argument values are read on entry, before the `-O0` prologue stores the
incoming arguments to their stack slots, so they are correct at any
optimization level; return values likewise.

### Consumers

`compiler-rt` defines both callbacks weakly and empty in `sanitizer_common`, so
a program compiled with `trace-args`/`trace-ret` links without a runtime that
consumes the values; a strong definition overrides the default.

The Linux kernel consumes the same callbacks through the KCOV dataflow
subsystem rather than through compiler-rt.

### Supported scope

The two flags are validated for:

* C and C++ compiled by Clang, and Rust compiled by a `rustc` whose LLVM
  carries the pass.
* ELF targets on x86-64 and AArch64, both in userspace (a custom runtime) and
  in the Linux kernel.
* Code built with or without `-g`, at any optimization level.

Any other frontend or target gets valid instrumentation: the pass works on
LLVM IR and target-independent IR attributes only, and never emits IR the
verifier rejects - a Swift `swifterror` slot, for one, is not reported.

## Tracing control flow

With `-fsanitize-coverage=control-flow` the compiler will create a table to collect
control flow for each function. More specifically, for each basic block in the function,
two lists are populated. One list for successors of the basic block and another list for
non-intrinsic called functions.

**TODO:** in the current implementation, indirect calls are not tracked
and are only marked with special value (-1) in the list.

Each table row consists of the basic block address
followed by `null`-ended lists of successors and callees.
The table is encoded in a special section named `sancov_cfs`

Example:

```c++
int foo (int x) {
  if (x > 0)
    bar(x);
  else
    x = 0;
  return x;
}
```

The code above contains 4 basic blocks, let's name them A, B, C, D:

```none
A
|\
| \
B  C
| /
|/
D
```

The collected control flow table is as follows:
`A, B, C, null, null, B, D, null, @bar, null, C, D, null, null, D, null, null.`

Users need to implement a single function to capture the CF table at startup:

```c++
extern "C"
void __sanitizer_cov_cfs_init(const uintptr_t *cfs_beg,
                              const uintptr_t *cfs_end) {
  // [cfs_beg,cfs_end) is the array of ptr-sized integers representing
  // the collected control flow.
}
```

## Tracing Stack Depth

With `-fsanitize-coverage=stack-depth` the compiler will track how much
stack space has been used for a function call chain. Leaf functions are
not included in this tracing.

The maximum depth of a function call graph is stored in the thread-local
`__sancov_lowest_stack` variable. Instrumentation is inserted in every
non-leaf function to check the frame pointer against this variable,
and if it is lower, store the current frame pointer. This effectively
inserts the following:

```c++
extern thread_local uintptr_t __sancov_lowest_stack;

uintptr_t stack = (uintptr_t)__builtin_frame_address(0);
if (stack < __sancov_lowest_stack)
  __sancov_lowest_stack = stack;
```

If `-fsanitize-coverage-stack-depth-callback-min=N` (where
`N > 0`) is also used, the tracking is delegated to a callback,
`__sanitizer_cov_stack_depth`, instead of adding instrumentation to
update `__sancov_lowest_stack`. The `N` of the argument is used
to determine which functions to instrument. Only functions estimated
to be using `N` bytes or more of stack space will be instrumented to
call the tracing callback. In the case of a dynamically sized stack,
the callback is unconditionally added.

The callback takes no arguments and is responsible for determining
the stack usage and doing any needed comparisons and storage. A roughly
equivalent implementation of `__sancov_lowest_stack` using the callback
would look like this:

```c++
void __sanitizer_cov_stack_depth(void) {
  uintptr_t stack = (uintptr_t)__builtin_frame_address(0);

  if (stack < __sancov_lowest_stack)
    __sancov_lowest_stack = stack;
}
```

## Gated Trace Callbacks

Gate the invocation of the tracing callbacks with
`-sanitizer-coverage-gated-trace-callbacks`.

When this option is enabled, the instrumentation will not call into the
runtime-provided callbacks for tracing, thus only incurring in a trivial
branch without going through a function call.

It is up to the runtime to toggle the value of the global variable in order to
enable tracing.

This option is only supported for trace-pc-guard and trace-cmp.

## Disabling instrumentation with `__attribute__((no_sanitize("coverage")))`

It is possible to disable coverage instrumentation for select functions via the
function attribute `__attribute__((no_sanitize("coverage")))`. Because this
attribute may not be supported by other compilers, it is recommended to use it
together with `__has_feature(coverage_sanitizer)`.

## Disabling instrumentation without source modification

It is sometimes useful to tell SanitizerCoverage to instrument only a subset of the
functions in your target without modifying source files.
With `-fsanitize-coverage-allowlist=allowlist.txt`
and `-fsanitize-coverage-ignorelist=blocklist.txt`,
you can specify such a subset through the combination of an allowlist and a blocklist.

SanitizerCoverage will only instrument functions that satisfy two conditions.
First, the function should belong to a source file with a path that is both allowlisted
and not blocklisted.
Second, the function should have a mangled name that is both allowlisted and not blocklisted.

The allowlist and blocklist format is similar to that of the sanitizer blocklist format.
The default allowlist will match every source file and every function.
The default blocklist will match no source file and no function.

A common use case is to have the allowlist list folders or source files for which you want
instrumentation and allow all function names, while the blocklist will opt out some specific
files or functions that the allowlist loosely allowed.

Here is an example allowlist:

```none
# Enable instrumentation for a whole folder
src:bar/*
# Enable instrumentation for a specific source file
src:foo/a.cpp
# Enable instrumentation for all functions in those files
fun:*
```

And an example blocklist:

```none
# Disable instrumentation for a specific source file that the allowlist allowed
src:bar/b.cpp
# Disable instrumentation for a specific function that the allowlist allowed
fun:*myFunc*
```

The use of `*` wildcards above is required because function names are matched after mangling.
Without the wildcards, one would have to write the whole mangled name.

Be careful that the paths of source files are matched exactly as they are provided on the clang
command line.
For example, the allowlist above would include file `bar/b.cpp` if the path was provided
exactly like this, but would it would fail to include it with other ways to refer to the same
file such as `./bar/b.cpp`, or `bar\b.cpp` on Windows.
So, please make sure to always double check that your lists are correctly applied.

## Default implementation

The sanitizer run-time (AddressSanitizer, MemorySanitizer, etc) provide a
default implementations of some of the coverage callbacks.
You may use this implementation to dump the coverage on disk at the process
exit.

Example:

```console
% cat -n cov.cc
     1  #include <stdio.h>
     2  __attribute__((noinline))
     3  void foo() { printf("foo\n"); }
     4
     5  int main(int argc, char **argv) {
     6    if (argc == 2)
     7      foo();
     8    printf("main\n");
     9  }
% clang++ -g cov.cc -fsanitize=address -fsanitize-coverage=trace-pc-guard
% ASAN_OPTIONS=coverage=1 ./a.out; wc -c *.sancov
main
SanitizerCoverage: ./a.out.7312.sancov 2 PCs written
24 a.out.7312.sancov
% ASAN_OPTIONS=coverage=1 ./a.out foo ; wc -c *.sancov
foo
main
SanitizerCoverage: ./a.out.7316.sancov 3 PCs written
24 a.out.7312.sancov
32 a.out.7316.sancov
```

Set `print_coverage_summary=0` (for example,
`ASAN_OPTIONS=coverage=1:print_coverage_summary=0`) to suppress the
`SanitizerCoverage:` summary line. Coverage files are still written.

Every time you run an executable instrumented with SanitizerCoverage
one `*.sancov` file is created during the process shutdown.
If the executable is dynamically linked against instrumented DSOs,
one `*.sancov` file will be also created for every DSO.

### Sancov data format

The format of `*.sancov` files is very simple: the first 8 bytes is the magic,
one of `0xC0BFFFFFFFFFFF64` and `0xC0BFFFFFFFFFFF32`. The last byte of the
magic defines the size of the following offsets. The rest of the data is the
offsets in the corresponding binary/DSO that were executed during the run.

### Sancov Tool

A simple `sancov` tool is provided to process coverage files.
The tool is part of LLVM project and is currently supported only on Linux.
It can handle symbolization tasks autonomously without any extra support
from the environment. You need to pass .sancov files (named
`<module_name>.<pid>.sancov` and paths to all corresponding binary elf files.
Sancov matches these files using module names and binaries file names.

```console
USAGE: sancov [options] <action> <binary files...> <.sancov files...> <.symcov files...>

Action (required):
  -covered-functions     Print all covered funcions.
  -diff                  Compute difference between two sancov files (A - B) and write to the new output sancov file
  -html-report           REMOVED. Use -symbolize & coverage-report-server.py.
  -merge                 Merges reports.
  -not-covered-functions Print all not covered funcions.
  -print-coverage-pcs    Print coverage instrumentation points addresses.
  -print-coverage-stats  Print coverage statistics.
  -print                 Print coverage addresses
  -symbolize             Produces a symbolized JSON report from binary report.
  -union                 Compute union of multiple sancov files and write to the new output sancov file

Generic Options:
  -help    Display this help
  -h       Alias for --help
  -version Display the version
  -v       Alias for --version

OPTIONS:
  -demangle=0          Alias for --no-demangle
  -demangle            Demangle function names
  -ignorelist=<string> Ignorelist file (sanitizer ignorelist format)
  -no-demangle         Do not demangle function names
  -no-skip-dead-files  List dead source files in reports
  -output=<string>     Output file for diff and union actions
  -skip-dead-files=0   Alias for --no-skip-dead-files
  -skip-dead-files     Do not list dead source files in reports
  -strip_path_prefix=<string>
                      Strip this prefix from files paths in reports
  -use_default_ignorelist=0
                      Alias for --no-use_default_ignore_list
```

### Coverage Reports

**Experimental**

`.sancov` files do not contain enough information to generate a source-level
coverage report. The missing information is contained
in debug info of the binary. Thus the `.sancov` has to be symbolized
to produce a `.symcov` file first:

```console
sancov -symbolize my_program.123.sancov my_program > my_program.123.symcov
```

The `.symcov` file can be browsed overlaid over the source code by
running `tools/sancov/coverage-report-server.py` script that will start
an HTTP server.

### Output directory

By default, .sancov files are created in the current working directory.
This can be changed with `ASAN_OPTIONS=coverage_dir=/path`:

```console
% ASAN_OPTIONS="coverage=1:coverage_dir=/tmp/cov" ./a.out foo
% ls -l /tmp/cov/*sancov
-rw-r----- 1 kcc eng 4 Nov 27 12:21 a.out.22673.sancov
-rw-r----- 1 kcc eng 8 Nov 27 12:21 a.out.22679.sancov
```
