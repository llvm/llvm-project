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

See [OpenACC-extensions.md](OpenACC-extensions.md) for the full list of
extensions supported by Flang. Flang also adds the following restrictions:

* The OpenACC specification does not prohibit the same variable from appearing
  in multiple data clauses, but this is disallowed for variables appearing in
  `reduction` clauses.
* The OpenACC specification does not prohibit the same variable from appearing
  multiple times in a `use_device` clause on a `host_data` construct, but this
  is disallowed.

## Remarks about incompatibilities with other implementations
* Array element references in the data clauses are equivalent to array sections
  consisting of this single element, i.e. `copyin(a(n))` is equivalent to
  `copyin(a(n:n))`.  Some other implementations have treated it as
  `copyin(a(:n))`, which does not correspond to OpenACC spec – Flang does not
  support this interpretation of an array element reference.
