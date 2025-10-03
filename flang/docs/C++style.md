<!--===- docs/C++style.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Flang C++ Style Guide

```{contents}
---
local:
---
```

This document captures the style guide rules that are followed in the Flang codebase.

## In brief:
* Use *clang-format*
from llvm 7
on all C++ source and header files before
every merge to main.  All code layout should be determined
by means of clang-format.
* Where a clear precedent exists in the project, follow it.
* Otherwise, where [LLVM's C++ style guide](https://llvm.org/docs/CodingStandards.html#style-issues)
is clear on usage, follow it.
* Otherwise, where a good public C++ style guide is relevant and clear,
  follow it.  [Google's](https://google.github.io/styleguide/cppguide.html)
  is pretty good and comes with lots of justifications for its rules.
* Reasonable exceptions to these guidelines can be made.
* Be aware of some workarounds for known issues in older C++ compilers that should
  still be able to compile f18. They are listed at the end of this document.

## In particular:

Use serial commas in comments, error messages, and documentation
unless they introduce ambiguity.

### Error messages
1. Messages should be a single sentence with few exceptions.
1. Fortran keywords should appear in upper case.
1. Names from the program appear in single quotes.
1. Messages should start with a capital letter.
1. Messages should not end with a period.

### Files
1. File names should use dashes, not underscores.  C++ sources have the
extension ".cpp", not ".C" or ".cc" or ".cxx".  Don't create needless
source directory hierarchies.
1. Header files should be idempotent.  Use the usual technique:
```
#ifndef FORTRAN_header_H_
#define FORTRAN_header_H_
// code
#endif  // FORTRAN_header_H_
```
1. `#include` every header defining an entity that your project header or source
file actually uses directly.  (Exception: when foo.cpp starts, as it should,
with `#include "foo.h"`, and foo.h includes bar.h in order to define the
interface to the module foo, you don't have to redundantly `#include "bar.h"`
in foo.cpp.)
1. In the source file "foo.cpp", put its corresponding `#include "foo.h"`
first in the sequence of inclusions.
Then `#include` other project headers in alphabetic order; then C++ standard
headers, also alphabetically; then C and system headers.
1. Don't use `#include <iostream>`.  If you need it for temporary debugging,
remove the inclusion before committing.

### Naming
1. C++ names that correspond to well-known interfaces from the STL, LLVM,
and Fortran standard
can and should look like their models when the reader can safely assume that
they mean the same thing -- e.g., `clear()` and `size()` member functions
in a class that implements an STL-ish container.
Fortran intrinsic function names are conventionally in ALL CAPS.
1. Non-public data members should be named with leading miniscule (lower-case)
letters, internal camelCase capitalization, and a trailing underscore,
e.g. `DoubleEntryBookkeepingSystem myLedger_;`.  POD structures with
only public data members shouldn't use trailing underscores, since they
don't have class functions from which data members need to be distinguishable.
1. Accessor member functions are named with the non-public data member's name,
less the trailing underscore.  Mutator member functions are named `set_...`
and should return `*this`.  Don't define accessors or mutators needlessly.
1. Other class functions should be named with leading capital letters,
CamelCase, and no underscores, and, like all functions, should be based
on imperative verbs, e.g. `HaltAndCatchFire()`.
1. It is fine to use short names for local variables with limited scopes,
especially when you can declare them directly in a `for()`/`while()`/`if()`
condition.  Otherwise, prefer complete English words to abbreviations
when creating names.

### Commentary
1. Use `//` for all comments except for short `/*notes*/` within expressions.
1. When `//` follows code on a line, precede it with two spaces.
1. Comments should matter.  Assume that the reader knows current C++ at least as
well as you do and avoid distracting her by calling out usage of new
features in comments.

### Layout
Always run `clang-format` on your changes before committing code. LLVM
has a `git-clang-format` script to facilitate running clang-format only
on the lines that have changed.

Here's what you can expect to see `clang-format` do:
1. Indent with two spaces.
1. Don't indent public:, protected:, and private:
accessibility labels.
1. Never use more than 80 characters per source line.
1. Don't use tabs.
1. Don't indent the bodies of namespaces, even when nested.
1. Function result types go on the same line as the function and argument
names.

Don't try to make columns of variable names or comments
align vertically -- they are maintenance problems.

Always wrap the bodies of `if()`, `else`, `while()`, `for()`, `do`, &c.
with braces, even when the body is a single statement or empty.  Note that this
diverges from the LLVM coding style.  In parts of the codebase that make heavy
use of LLVM or MLIR APIs (e.g. the Lower and Optimizer libraries), use the
LLVM style instead.  The
opening `{` goes on
the end of the line, not on the next line.  Functions also put the opening
`{` after the formal arguments or new-style result type, not on the next
line.  Use `{}` for empty inline constructors and destructors in classes.

If any branch of an `if`/`else if`/`else` cascade ends with a return statement,
they all should, with the understanding that the cases are all unexceptional.
When testing for an error case that should cause an early return, do so with
an `if` that doesn't have a following `else`.

Don't waste space on the screen with needless blank lines or elaborate block
commentary (lines of dashes, boxes of asterisks, &c.).  Write code so as to be
easily read and understood with a minimum of scrolling.

Avoid using assignments in controlling expressions of `if()` &c., even with
the idiom of wrapping them with extra parentheses.

In multi-element initializer lists (especially `common::visitors{...}`),
including a comma after the last element often causes `clang-format` to do
a better jobs of formatting.

### C++ language
Use *C++17*, unless some compiler to which we must be portable lacks a feature
you are considering.
However:
1. Never throw or catch exceptions.
1. Never use run-time type information or `dynamic_cast<>`.
1. Never declare static data that executes a constructor.
   (This is why `#include <iostream>` is contraindicated.)
1. Use `{braced initializers}` in all circumstances where they work, including
default data member initialization.  They inhibit implicit truncation.
Don't use `= expr` initialization just to effect implicit truncation;
prefer an explicit `static_cast<>`.
With C++17, braced initializers work fine with `auto` too.
Sometimes, however, there are better alternatives to empty braces;
e.g., prefer `return std::nullopt;` to `return {};` to make it more clear
that the function's result type is a `std::optional<>`.
1. Avoid unsigned types apart from `size_t`, which must be used with care.
When `int` just obviously works, just use `int`.  When you need something
bigger than `int`, use `std::int64_t` rather than `long` or `long long`.
1. Use namespaces to avoid conflicts with client code.  Use one top-level
`Fortran` project namespace.  Don't introduce needless nested namespaces within the
project when names don't conflict or better solutions exist.  Never use
`using namespace ...;` outside test code; never use `using namespace std;`
anywhere.  Access STL entities with names like `std::unique_ptr<>`,
without a leading `::`.
1. Prefer `static` functions over functions in anonymous namespaces in source files.
1. Use `auto` judiciously.  When the type of a local variable is known,
monomorphic, and easy to type, be explicit rather than using `auto`.
Don't use `auto` functions unless the type of the result of an outlined member
function definition can be more clear due to its use of types declared in the
class.
1. Use move semantics and smart pointers to make dynamic memory ownership
clear.  Consider reworking any code that uses `malloc()` or a (non-placement)
`operator new`.
See the section on Pointers below for some suggested options.
1. When defining argument types, use values when object semantics are
not required and the value is small and copyable without allocation
(e.g., `int`);
use `const` or rvalue references for larger values (e.g., `std::string`);
use `const` references to rather than pointers to immutable objects;
and use non-`const` references for mutable objects, including "output" arguments
when they can't be function results.
Put such output arguments last (_pace_ the standard C library conventions for `memcpy()` & al.).
1. Prefer `typename` to `class` in template argument declarations.
1. Prefer `enum class` to plain `enum` wherever `enum class` will work.
We have an `ENUM_CLASS` macro that helps capture the names of constants.
1. Use `constexpr` and `const` generously.
1. When a `switch()` statement's labels do not cover all possible case values
explicitly, it should contain either a `default:;` at its end or a
`default:` label that obviously crashes; we have a `CRASH_NO_CASE` macro
for such situations.
1. On the other hand, when a `switch()` statement really does cover all of
the values of an `enum class`, please insert a call to the `SWITCH_COVERS_ALL_CASES`
macro at the top of the block.  This macro does the right thing for G++ and
clang to ensure that no warning is emitted when the cases are indeed all covered.
1. When using `std::optional` values, avoid unprotected access to their content.
This is usually by means of `x.has_value()` guarding execution of `*x`.
This is implicit when they are function results assigned to local variables
in `if`/`while` predicates.
When no presence test is obviously protecting a `*x` reference to the
contents, and it is assumed that the contents are present, validate that
assumption by using `x.value()` instead.
1. We use `c_str()` rather than `data()` when converting a `std::string`
to a `const char *` when the result is expected to be NUL-terminated.
1. Avoid explicit comparisons of pointers to `nullptr` and tests of
presence of `optional<>` values with `.has_value()` in the predicate
expressions of control flow statements, but prefer them to implicit
conversions to `bool` when initializing `bool` variables and arguments,
and to the use of the idiom `!!`.
(But please use `.has_value()` or `.value_or()` with `optional<bool>`
to avoid a common pitfall or the appearance of having fallen into it.)

#### Classes
1. Define POD structures with `struct`.
1. Don't use `this->` in (non-static) member functions, unless forced to
do so in a template member function.
1. Define accessor and mutator member functions (implicitly) inline in the
class, after constructors and assignments.  Don't needlessly define
(implicit) inline member functions in classes unless they really solve a
performance problem.
1. Try to make class definitions in headers concise specifications of
interfaces, at least to the extent that C++ allows.
1. When copy constructors and copy assignment are not necessary,
and move constructors/assignment is present, don't declare them and they
will be implicitly deleted.  When neither copy nor move constructors
or assignments should exist for a class, explicitly `=delete` all of them.
1. Make single-argument constructors (other than copy and move constructors)
'explicit' unless you really want to define an implicit conversion.

#### Pointers
There are many -- perhaps too many -- means of indirect addressing
data in this project.
Some of these are standard C++ language and library features,
while others are local inventions in `lib/Common`:
* Bare pointers (`Foo *p`): these are obviously nullable, non-owning,
undefined when uninitialized, shallowly copyable, reassignable, and often
not the right abstraction to use in this project.
But they can be the right choice to represent an optional
non-owning reference, as in a function result.
Use the `DEREF()` macro to convert a pointer to a reference that isn't
already protected by an explicit test for null.
* References (`Foo &r`, `const Foo &r`): non-nullable, not owning,
shallowly copyable, and not reassignable.
References are great for invisible indirection to objects whose lifetimes are
broader than that of the reference.
Take care when initializing a reference with another reference to ensure
that a copy is not made because only one of the references is `const`;
this is a pernicious C++ language pitfall!
* Rvalue references (`Foo &&r`): These are non-nullable references
*with* ownership, and they are ubiquitously used for formal arguments
wherever appropriate.
* `std::reference_wrapper<>`: non-nullable, not owning, shallowly
copyable, and (unlike bare references) reassignable, so suitable for
use in STL containers and for data members in classes that need to be
copyable or assignable.
* `common::Reference<>`: like `std::reference_wrapper<>`, but also supports
move semantics, member access, and comparison for equality; suitable for use in
`std::variant<>`.
* `std::unique_ptr<>`: A nullable pointer with ownership, null by default,
not copyable, reassignable.
F18 has a helpful `Deleter<>` class template that makes `unique_ptr<>`
easier to use with forward-referenced data types.
* `std::shared_ptr<>`: A nullable pointer with shared ownership via reference
counting, null by default, shallowly copyable, reassignable, and slow.
* `Indirection<>`: A non-nullable pointer with ownership and
optional deep copy semantics; reassignable.
Often better than a reference (due to ownership) or `std::unique_ptr<>`
(due to non-nullability and copyability).
Can be wrapped in `std::optional<>` when nullability is required.
Usable with forward-referenced data types with some use of `extern template`
in headers and explicit template instantiation in source files.
* `CountedReference<>`: A nullable pointer with shared ownership via
reference counting, null by default, shallowly copyable, reassignable.
Safe to use *only* when the data are private to just one
thread of execution.
Used sparingly in place of `std::shared_ptr<>` only when the overhead
of that standard feature is prohibitive.

A feature matrix:

| indirection           | nullable | default null | owning | reassignable | copyable          | undefined type ok? |
| -----------           | -------- | ------------ | ------ | ------------ | --------          | ------------------ |
| `*p`                  | yes      | no           | no     | yes          | shallowly         | yes                |
| `&r`                  | no       | n/a          | no     | no           | shallowly         | yes                |
| `&&r`                 | no       | n/a          | yes    | no           | shallowly         | yes                |
| `reference_wrapper<>` | no       | n/a          | no     | yes          | shallowly         | yes                |
| `Reference<>`         | no       | n/a          | no     | yes          | shallowly         | yes                |
| `unique_ptr<>`        | yes      | yes          | yes    | yes          | no                | yes, with work     |
| `shared_ptr<>`        | yes      | yes          | yes    | yes          | shallowly         | no                 |
| `Indirection<>`       | no       | n/a          | yes    | yes          | optionally deeply | yes, with work     |
| `CountedReference<>`  | yes      | yes          | yes    | yes          | shallowly         | no                 |

### Overall design preferences
Don't use dynamic solutions to solve problems that can be solved at
build time; don't solve build time problems by writing programs that
produce source code when macros and templates suffice; don't write macros
when templates suffice.  Templates are statically typed, checked by the
compiler, and are (or should be) visible to debuggers.

### Exceptions to these guidelines
Reasonable exceptions will be allowed; these guidelines cannot anticipate
all situations.
For example, names that come from other sources might be more clear if
their original spellings are preserved rather than mangled to conform
needlessly to the conventions here, as Google's C++ style guide does
in a way that leads to weirdly capitalized abbreviations in names
like `Http`.
Consistency is one of many aspects in the pursuit of clarity,
but not an end in itself.

## C++ compiler bug workarounds
Below is a list of workarounds for C++ compiler bugs met with f18 that, even
if the bugs are fixed in latest C++ compiler versions, need to be applied so
that all desired tool-chains can compile f18.

### Explicitly move noncopyable local variable into optional results

The following code is legal C++ but fails to compile with the
default Ubuntu 18.04 g++ compiler (7.4.0-1ubuntu1~18.0.4.1):

```
class CantBeCopied {
 public:
 CantBeCopied(const CantBeCopied&) = delete;
 CantBeCopied(CantBeCopied&&) = default;
 CantBeCopied() {}
};
std::optional<CantBeCopied> fooNOK() {
  CantBeCopied result;
  return result; // Legal C++, but does not compile with Ubuntu 18.04 default g++
}
std::optional<CantBeCopied> fooOK() {
  CantBeCopied result;
  return {std::move(result)}; // Compiles OK everywhere
}
```
The underlying bug is actually not specific to `std::optional` but this is the most common
case in f18 where the issue may occur. The actual bug can be reproduced with any class `B`
that has a perfect forwarding constructor taking `CantBeCopied` as argument:
`template<typename CantBeCopied> B(CantBeCopied&& x) x_{std::forward<CantBeCopied>(x)} {}`.
In such scenarios, Ubuntu 18.04 g++ fails to instantiate the move constructor
and to construct the returned value as it should, instead it complains about a
missing copy constructor.

Local result variables do not need to and should not be explicitly moved into optionals
if they have a copy constructor.
