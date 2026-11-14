(contributing)=

# Contributing to the libc Project

LLVM-libc is being developed as part of the LLVM project so contributions
to the libc project should also follow the general LLVM
[contribution guidelines](https://llvm.org/docs/Contributing.html). Below is
a list of open projects that one can start with:

1. **Beginner Bugs** - Help us tackle
   [good first issues](https://github.com/llvm/llvm-project/issues?q=is%3Aopen+is%3Aissue+label%3Alibc+label%3A%22good+first+issue%22).
   These bugs have been tagged with the github labels "libc" and "good first
   issue" by the team as potentially easier places to get started. Please do
   first check if the bug has an assignee; if so please find another unless
   there's been no movement on the issue from the assignee, in which place do
   ask if you can help take over. Remember to follow the LLVM
   [AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html) which forbids
   using LLMs for good first issues.
2. **Cleanup code-style** - The libc project follows the general
   [LLVM style](https://llvm.org/docs/CodingStandards.html) with specific
   conventions for naming (`snake_case` for functions, `CamelCase` for
   types). See the {doc}`code style <dev/code_style>` page for the
   authoritative reference.
   Mechanical projects to move parts following old styles to the current
   conventions are welcome.
3. **Implement Linux syscall wrappers** - A large portion of the POSIX API can
   be implemented as syscall wrappers on Linux. A good number have already been
   implemented but many more are yet to be implemented. So, a project of medium
   complexity would be to implement syscall wrappers which have not yet been
   implemented.
4. **Update the clang-tidy lint rules and use them in the build and/or CI** -
   The libc project has a set of clang-tidy checks (see
   {ref}`clang-tidy checks <clang_tidy_checks>`)
   but they are not enabled by default. They can be enabled by configuring with
   `-DLLVM_LIBC_ENABLE_LINTING=ON` (or by setting `LLVM_LIBC_CLANG_TIDY`) and
   running the `libc-lint` build target. This project is about keeping the
   checks up to date and reintegrating them into the build and CI.
5. **double and higher precision math functions** - These are under active
   development but you can take a shot at those not yet implemented. See
   {ref}`math` for more information.
6. **Contribute a new OS/Architecture port** - You can contribute a new
   operating system or target architecture port. See {ref}`porting` for more
   information.

## Before you start

Read {ref}`design_philosophy` and {ref}`abi_policy` to understand what the
project is and is not trying to do. Read {ref}`patterns_to_avoid` before
modelling new code on something that is already in the tree, because older
code does not always follow the current conventions.

Some changes are worth discussing with the maintainers before investing in an
implementation:

- a non-trivial change to the layout of an existing public type or the value
  of an existing public macro, or exporting a symbol under a name other than
  its standard one (see {ref}`abi_policy`),
- a function that is not in the C or POSIX standards,
- a new configuration option,
- new code under `libc/shared/`, or a new user of it,
- any code written in assembly,
- any intentional deviation from a standard.

The easiest way to reach out is to ask in the `#libc` channel on the
[LLVM Discord](https://discord.gg/xS7Z362).

## Pull requests

- **Keep pull requests small and focused.** Add one function, or one closely
  related family of functions, per pull request. Land refactoring separately
  from the change that needs it. A bug found along the way gets its own pull
  request.
- **Title.** Start the title with `[libc]`, and add a second tag where one
  applies: `[libc][math]`, `[libc][docs]`, or `[libc][NFC]` for changes with
  no functional effect.
- **Format the change.** Run `clang-format` on C++ sources and
  `libc/utils/cmake_format.py -i` on the CMake files you edited.
- **Test in both build modes** where the code supports both. See
  {ref}`testing_guide`.
- **Do not mention people with `@` in commit messages**, or in a pull request
  description that will become one. GitHub notifies them again each time the
  commit is pushed to another fork. Request reviews through the reviewers
  field instead.
- **Bazel.** Updating the Bazel build files is not required, but it is
  appreciated.

LLVM-libc follows LLVM's
[patch reversion policy](https://llvm.org/docs/DeveloperPolicy.html#patch-reversion-policy).
A change that breaks a buildbot is reverted or, if the fix is trivial, fixed
forward. When relanding, describe what went wrong and how it was fixed.

## AI-assisted contributions

LLVM's [AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html) applies
to LLVM-libc. In short: a person must read, understand, and be able to answer
questions about everything they submit; contributions with substantial
tool-generated content are labelled, for example with an `Assisted-by:`
trailer; the policy strongly recommends writing the pull request description
yourself; and AI tools must not be used on issues labelled "good first issue".

Tools make some mistakes in this project more often than people do:

- They imitate the code around them, including code that follows outdated
  conventions. See {ref}`patterns_to_avoid`.
- Asked to make a program work, they change LLVM-libc's types, symbols, or
  behavior to match another libc, because that is the shortest path to a
  passing build. See {ref}`making_existing_software_work`.
- They reproduce code from other C libraries. That is not allowed, regardless
  of how the code was produced.
- They call one public function from another. See {ref}`code_organization`.

Check for these before asking for review.
