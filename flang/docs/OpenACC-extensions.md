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
* `!$acc cache` directives accept scalar variables.
* `!$acc cache` directives are accepted outside of a loop construct.
* The `!$acc declare` directive accepts assumed-size array arguments for
  `deviceptr` and `present` clauses.
* The OpenACC specification disallows a variable from appearing multiple times
  in clauses of `!$acc declare` directives for a function, subroutine, program,
  or module, but Flang permits it with a warning when the same clause is used.

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

## Extensions enabled by flag

### `-fno-openacc-default-none-scalars-strict` — pre-OpenACC-3.2 scalar behavior under `DEFAULT(NONE)`

OpenACC version 3.2 (section 1.16, change 733) clarified that the
`default(none)` clause applies to scalar variables.  Prior to version 3.2,
`default(none)` did not impose a data-clause requirement on scalar variables.

By default, Flang enforces the OpenACC 3.2 behavior: scalar variables
referenced inside a `default(none)` compute region without an explicit data
clause produce an error.

When `-fno-openacc-default-none-scalars-strict` is specified, Flang reverts to
the pre-3.2 behavior: scalar variables referenced inside a `default(none)`
compute region without an explicit data clause do not produce an error.
Instead, Flang infers implicit data attributes for those scalars via the same
implicit-copy logic applied in regions without `default(none)`.

Array variables always require an explicit data clause under `default(none)`
regardless of this flag.

When a scalar is implicitly attributed under this extension no warning is
emitted by default; explicit opt-in to the non-standard behavior is treated as
acknowledgement.  Use `-Wopenacc-default-none-scalars-strict` to enable
per-use warnings for audit purposes.
