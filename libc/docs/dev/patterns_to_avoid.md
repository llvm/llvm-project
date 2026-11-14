(patterns_to_avoid)=

# Common Mistakes and Legacy Patterns

LLVM-libc's conventions have changed over the years, and the tree still
contains code written to older ones. **Existing code is not precedent.** A
pattern being present in the tree, even many times, does not make it
acceptable in a new change. This page lists the mistakes reviewers see most
often and the old patterns that should not be copied.

This applies with particular force to changes written with AI assistance,
because a tool that imitates nearby code will imitate its outdated parts too.

## Most common mistakes

1. **Calling one entrypoint from another.** It is the natural way to write
   many functions and it is not allowed. See {ref}`code_organization`.
2. **Mimicking another libc's interface.** Changing a type's layout, a
   function's observable behavior, or the set of exported symbols solely to
   match another libc is sometimes the quickest way to get an existing program
   running. That matching is not the intent for LLVM-libc, and goes against its
   design principles. See: {ref}`making_existing_software_work`.
3. **Copying an outdated pattern from old code.** See the table below.
4. **Copying from another libc.** This is not allowed for licensing reasons,
   and translated C does not fit this code base. See
   {ref}`design_philosophy`.
5. **Leaving out part of the registration.** A new function needs an entry in
   `libc/include/<header>.yaml` and in the relevant `entrypoints.txt` files in
   addition to its implementation. See {ref}`implementing_a_function`.

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
- **`while (true)` with a single `break`.** The condition that triggers the
  `break` is the loop condition. Write it there.
- **Unnecessary lambdas.** Use a lambda when a callable is needed. For
  ordinary code reuse, write a function.
- **Templates and variadic forwarding where a fixed signature would do.**
- **Magic constants.** A bare `7`, `61`, or `0xFFF` in an expression leaves
  the reader to work out what it stands for. Give every such constant a name
  that says what it means. Multiplying by `DAYS_PER_WEEK` explains itself;
  multiplying by `7` does not.
