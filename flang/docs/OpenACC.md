<!--===- docs/Extensions.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# OpenACC in Flang

```eval_rst
.. contents::
   :local:
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
