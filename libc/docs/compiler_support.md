(compiler_support)=

# Compiler Support

`LLVM libc` compiles from both `Clang` and `GCC` but for maximum
performance we recommend using `Clang`.

Indeed, some memory function implementations rely on [compiler intrinsics](https://clang.llvm.org/docs/LanguageExtensions.html#guaranteed-inlined-copy)
that are not currently available in `GCC`.
As such we cannot guarantee optimal performance for these functions.

For platforms where only `GCC` is natively available but maximum performance
is required it is possible to bootstrap `Clang` with `GCC` and then use
`Clang` to build the `libc` project.

## Minimum supported versions

- `Clang 11`
- `GCC 12.2`

Code that is shared with other LLVM projects through
{ref}`Hand-in-Hand <hand_in_hand>` is compiled as part of those projects, so
it must also build with the
[host compilers LLVM supports](https://llvm.org/docs/GettingStarted.html#host-c-toolchain-both-compiler-and-standard-library),
which are older.
