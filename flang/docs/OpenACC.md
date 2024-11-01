<!--===- docs/Extensions.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# OpenACC in Flang

```{contents}
---
local:
---
```

## Intentional deviation from the specification
* The end directive for combined construct can omit the `loop` keyword.
* An `!$acc routine` with no parallelism clause is treated as if the `seq`
  clause was present.
* `!$acc end loop` does not trigger a parsing error and is just ignored.
* The restriction on `!$acc data` required clauses is emitted as a portability
  warning instead of an error as other compiler accepts it.
* The `if` clause accepts scalar integer expression in addition to scalar
  logical expression.
* `!$acc routine` directive can be placed at the top level. 
* `!$acc cache` directive accepts scalar variable.

## Remarks about incompatibilities with other implementations
* Array element references in the data clauses are equivalent to array sections
  consisting of this single element, i.e. `copyin(a(n))` is equivalent to
  `copyin(a(n:n))`.  Some other implementations have treated it as
  `copyin(a(:n))`, which does not correspond to OpenACC spec – Flang does not
  support this interpretation of an array element reference.
