# C++ Profiles

```{contents}
:local:
```

## Introduction

**Experimental.** C++ profiles are being designed by WG21 (the framework is
[P3589](https://wg21.link/P3589)) and are not part of any published C++
standard. Clang's implementation is experimental, is enabled with the
`-cc1` option `-fexperimental-cxx-profiles`, and may change incompatibly as
the proposals evolve.

A profile is a named set of rules that a translation unit or module can ask to
have enforced. Code that violates an enforced rule is rejected. Profiles only
restrict which programs are accepted: they never change the meaning of a
program, overload resolution, template instantiation or SFINAE.

Clang currently implements the profile framework and one profile,
`std::casting`.

## Enforcing a profile

A profile is enforced by an attribute on an empty-declaration that precedes
every other declaration of the translation unit (including declarations from
included headers):

```c++
[[profiles::enforce(std::casting)]];
#include <vector>
```

or on a module declaration:

```c++
export module Widgets [[profiles::enforce(std::casting)]];
```

A profile enforced on the module declaration of a primary module interface is
also enforced in the implementation units of the module, and is advertised to
importers, which can require it:

```c++
import Widgets [[profiles::require(std::casting)]];
```

`require` does not enforce the profile in the importing translation unit; the
import is ill-formed if `Widgets` does not enforce `std::casting`.

Profiles can also be enforced for a whole translation unit from the command
line with `-fcxx-profiles=std::casting`. `-fcxx-profiles-preview=<profile>`
checks the rules of a profile but reports violations as warnings, which is
useful to evaluate a profile on existing code.

Code is checked according to the profiles enforced where it is written. In
particular, the instantiation of a template is checked according to the
profiles enforced by the translation unit or module that defines the template,
not by the one that instantiates it.

Code in system headers is never checked. Other headers can be exempted with
`[[profiles::exempt]]`, which exempts the named header and every header it
includes:

```c++
[[profiles::enforce(std::casting)]];
[[profiles::exempt(std::casting, quote_header: "legacy/ffi.h",
                   justification: "C interface")]];
#include "legacy/ffi.h"
```

The header is identified by matching the trailing components of its path.

## Suppressing rules

`[[profiles::suppress]]` turns off a profile, or one of its rules, for the
tokens of a declaration or statement. An optional justification documents why:

```c++
unsigned bits(float f) {
  [[profiles::suppress(std::casting, rule: reinterpret_cast,
                       justification: "inspecting the representation")]]
  return *reinterpret_cast<unsigned *>(&f);
}
```

## Diagnostics

Each rule of each profile is diagnosed in its own diagnostic group,
`-Wcxx-profile-<profile>-<rule>`, contained in `-Wcxx-profile-<profile>` and
`-Wcxx-profile`. Rule violations are errors by default. The diagnostics are only
emitted when the profile is enforced; enabling the warning groups does not
enforce a profile. Unknown profile and rule names are diagnosed with
`-Wunknown-cxx-profiles`.

## `std::casting`

`std::casting` rejects casts that cannot be verified to use objects according to
their types. It follows the casting rules of
[P3081R2](https://wg21.link/P3081) section 4 (Type.1 to Type.4); P3984 lists
"casting" as one of the profiles that make up type-and-resource safety. The rules
apply to named casts and equally to C-style and functional-style casts that
perform the same conversions; for those, a note offers to spell the cast as the
named cast it performs.

| Rule | Rejects |
|---|---|
| `reinterpret_cast` | `reinterpret_cast`, except to a pointer or reference to `std::byte`, from a pointer to an integer type that can hold any pointer, or a cast that only adds qualifiers |
| `reinterpret_cast_char` | `reinterpret_cast` to a pointer or reference to `char` or `unsigned char` |
| `const_cast` | casts that cast away constness (removing only `volatile` is permitted) |
| `static_downcast` | `static_cast` from a base class to a derived class (use `dynamic_cast`) |
| `static_cast_void` | `static_cast` from `void *` to a pointer to an object type other than `std::byte` |

## Current limitations

- Only `std::casting` is implemented.
- Profiles enforced by a module are not recorded for the global module fragment
  of that module, and a module partition implementation unit does not inherit
  the profiles enforced on the primary module interface.
- The compatibility of redeclarations across profiles (P3589
  [decl.profile.enforce]p5) is not checked.
- `[[profiles::exempt]]` matches headers by path suffix rather than by the
  spelling of the `#include` directive.
