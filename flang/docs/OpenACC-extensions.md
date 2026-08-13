<!--===- docs/OpenACC-extensions.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# OpenACC Extensions in Flang

```{contents}
---
local:
---
```

Flang is more lenient than the OpenACC specification requires for purposes of
compatibility. This document describes extensions to the OpenACC specification.
There are a couple of known places where the Flang compiler intentionally deviates
from the standard by being more strict than the specification; these are currently
listed in [OpenACC.md](OpenACC.md).

## Extensions always active

These extensions require no flag.

* The end directive for combined constructs can omit the `loop` keyword.
* An `!$acc routine` with no parallelism clause is treated as if the `seq`
  clause were present.
* `!$acc end loop` does not trigger a parsing error and is silently ignored.
* The restriction on required clauses for `!$acc data` is emitted as a
  portability warning rather than an error, matching the behavior of other
  compilers.
* The `if` clause accepts scalar integer expressions in addition to scalar
  logical expressions.
* `!$acc routine` directives can be placed at the top level.
* `!$acc routine` directives can be placed directly within an interface block
  (i.e. as an interface-specification, such as preceding the interface body they
  name). The OpenACC specification only permits the named `routine` directive in
  the specification part of a subroutine, function, or module, and the unnamed
  form within an interface body; Flang additionally accepts a `routine`
  directive between the `INTERFACE` statement and the interface bodies, applying
  a named directive to the interface body it names.
* `!$acc cache` directives accept scalar variables.
* `!$acc cache` directives are accepted outside of a loop construct.
* The `!$acc declare` directive accepts assumed-size array arguments for
  `deviceptr` and `present` clauses.
* The OpenACC specification disallows a variable from appearing multiple times
  in clauses of `!$acc declare` directives for a function, subroutine, program,
  or module, but Flang permits it with a warning when the same clause is used.
* The REDUCTION clause accepts a MINUS "-" operator which is not permitted in
  the OpenACC specification. A warning is issued for this use. 
* The `collapse` clause may be applied to a `DO CONCURRENT` loop, which the
  OpenACC specification does not permit. The collapse value must equal the
  number of `DO CONCURRENT` controls; the construct then lowers like the
  equivalent perfectly-nested `DO` loops. A portability warning is issued for
  this use (`-Wportability`, also enabled by `-pedantic`; suppress with
  `-Wno-portability`).

## Extensions enabled by default

### `-fopenacc-multiple-names-in-routine` — `!$acc routine(<name>[, <name>]*) <clause-list>`

The `ROUTINE` directive accepts a parenthesized list of more than one name
(e.g. `!$acc routine(foo, bar) seq`).  The OpenACC specification permits only a
single name; this extension is equivalent to writing one `ROUTINE` directive
per name, each with identical clauses.  A `BIND` clause may not be combined
with multiple names.  A warning is emitted for each such directive
(`-Wopenacc-multiple-names-in-routine`; suppress with
`-Wno-openacc-multiple-names-in-routine`).  This extension may be disabled with
`-fno-openacc-multiple-names-in-routine`.

### Pre-OpenACC-3.2 scalar behavior under `DEFAULT(NONE)`

OpenACC version 3.2 (section 1.16, change 733) clarified that the
`default(none)` clause applies to scalar variables.  Prior to version 3.2,
`default(none)` did not impose a data-clause requirement on scalar variables.

By default, Flang uses the pre-3.2 behavior: scalar variables referenced inside
a `default(none)` compute region without an explicit data clause do not produce
an error. Instead, Flang infers implicit data attributes for those scalars via
the same implicit-copy logic applied in regions without `default(none)`.

Array variables always require an explicit data clause under `default(none)`
regardless of this extension.

When a scalar is implicitly attributed under this extension, a warning is
emitted by default (`-Wopenacc-default-none-scalars-strict`; suppress with
`-Wno-openacc-default-none-scalars-strict`). Use
`-fopenacc-default-none-scalars-strict` to enforce the OpenACC 3.2 behavior and
produce an error for scalar variables that are not listed in an explicit data
clause. `-fno-openacc-default-none-scalars-strict` preserves the default
pre-3.2 behavior explicitly.
