(patterns_to_avoid)=

# Common Mistakes and Legacy Patterns

LLVM-libc's conventions have changed over the years, and the tree still
contains code written to older ones. **Existing code is not precedent.** A
pattern being present in the tree, even many times, does not make it
acceptable in a new change. This page lists the mistakes reviewers see most
often and the old patterns that should not be copied.

This applies with particular force to changes written with AI assistance,
because a tool that imitates nearby code will imitate its outdated parts too.

Larger areas of the tree that are known to need rework are listed in
{ref}`roadmap`. Do not model new code on them. Issues for LLVM-libc are filed
under the `libc` label in the
[issue tracker](https://github.com/llvm/llvm-project/labels/libc).

## Most common mistakes

1. **Calling one entrypoint from another.** It is the natural way to write
   many functions and it is not allowed. Some existing code does it; that code
   is to be cleaned up and is not an example to follow. See
   {ref}`entrypoints_do_not_call_entrypoints`.
2. **Mimicking another libc's interface.** Changing a type's layout, a
   function's observable behavior, or the set of exported symbols solely to
   match another libc is sometimes the quickest way to get an existing program
   running. That matching is not the intent for LLVM-libc, and goes against its
   design principles. See: {ref}`making_existing_software_work`.
3. **Copying an outdated pattern from old code.** See the table below.
4. **Copying or translating code from another libc.** Neither is allowed.
   {ref}`original_implementations` states where the line is between learning
   from another implementation and taking from it.
5. **Leaving out part of the registration.** A new function needs an entry in
   `libc/include/<header>.yaml` and in the relevant `entrypoints.txt` files in
   addition to its implementation. A new public type needs its own
   registration as well. See {ref}`implementing_a_function`.

## Legacy patterns

| Seen in the tree | Write this instead |
| --- | --- |
| A first line of the form `//===-- Title ---...---===//` in a file header | The current LLVM file header, shown in {ref}`implementation_guidelines`. Update the header of any file you touch. |
| `syscall_impl(...)` in an entrypoint, followed by a hand-written check of the result | A wrapper in `src/__support/OSUtil/<os>/syscall_wrappers/` that uses `syscall_checked`. See {ref}`syscall_wrapper_refactor`. |
| Functions and variables inside an anonymous namespace | `static` functions and variables. Anonymous namespaces are for types only, as in the [LLVM coding standards](https://llvm.org/docs/CodingStandards.html#restrict-visibility). |
| A `.cpp` file that gets its types from the implementation header it includes | Include what you use: each file includes the proxy header for every type and macro it names. |
| `#include <pthread.h>` or another public header in `libc/src/` | The proxy headers in `libc/hdr/`. |
| `libc_errno` assigned inside an internal helper | The helper returns `ErrorOr<T>` and the entrypoint assigns `libc_errno`. |
| A new file or directory named `utils` | A name that says what the code does. |
| `ASSERT_TRUE(a == b)` and `ASSERT_TRUE(p != nullptr)` in tests | `ASSERT_EQ(a, b)` and `ASSERT_NE(p, nullptr)`, which print both values on failure. |
| A bare path literal such as `"testdata/file.txt"` or `"/nonexistent"` in tests | `libc_make_test_file_path(...)`, including for paths that are meant not to exist. |

## Code that is hard to review

These are not style violations in the formatting sense. They make code hard to
reason about, and reviewers will ask for them to be rewritten.

- **Deeply nested control flow.** An `if` inside a loop inside a loop inside
  an `if` is difficult to follow. Collapse conditions, return early, and move
  inner blocks into helper functions.
- **`while (true)` where the exit test is the first or last statement of the
  body.** That test is the loop condition. Write it as a `while` or
  `do`/`while` condition. The same applies to a body that ends in
  `if (x) continue; else return;`. A loop whose exit test is in the middle of
  the body is a different case, and `while (true)` with a `break` is a
  reasonable way to write it.
- **Lambdas used as local helper functions.** A lambda that captures local
  variables has inputs that do not appear at the call site, so the call no
  longer shows what the helper reads or changes. If the code is only being
  factored out for reuse or readability, write a `static` function and pass
  what it needs explicitly. A short lambda that is passed to another function
  as a callable is fine.
- **Templates that accept more than the code supports.** A function template
  whose body only works for subclasses of one class, or for callables with
  one signature, should say so in its declaration, so that the compiler
  rejects an argument that makes no sense. Take the base class by reference,
  or take a `cpp::function_ref` with the expected signature, instead of an
  unconstrained `typename T`. Where a template is needed, constrain it, for
  example with `cpp::enable_if_t` or a `static_assert`. Overloading a function
  for the specific types it supports is fine, and so is a template that is
  generic on purpose. The same applies to variadic argument forwarding.
- **Magic constants.** A bare `7`, `61`, or `0xFFF` in an expression leaves
  the reader to work out what it stands for. Give every such constant a name
  that says what it means. Multiplying by `DAYS_PER_WEEK` explains itself;
  multiplying by `7` does not.
