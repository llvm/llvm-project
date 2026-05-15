<!--===- docs/OpenMP-extensions.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

(openmp-extensions)=
# OpenMP API Extensions Supported by Flang

See also {doc}`OpenMPSupport` for a general overview of OpenMP support in Flang.

```{contents}
---
local:
---
```

The Flang compiler supports several extensions to OpenMP API features, providing enhanced parallelism and data management capabilities for Fortran applications.  This document outlines the supported extensions and their usage within Flang.


## Supported OpenMP API Extensions

The following extensions are supported by Flang.


### ATOMIC Construct
The implementation of the ATOMIC construct follows OpenMP 6.0 with the following extensions:
- `x = x` is an allowed form of ATOMIC UPDATE.
  This is motivated by the fact that the equivalent forms `x = x+0` or `x = x*1` are allowed.
- Explicit type conversions are allowed in ATOMIC READ, WRITE or UPDATE constructs, and in the capture statement in ATOMIC UPDATE CAPTURE.
  The OpenMP spec requires intrinsic- or pointer-assignments, which include (as per the Fortran standard) implicit type conversions.  Since such conversions need to be handled, allowing explicit conversions comes at no extra cost.
- A literal `.true.` or `.false.` is an allowed condition in ATOMIC UPDATE COMPARE. [1]
- A logical variable is an allowed form of the condition even if its value is not computed within the ATOMIC UPDATE COMPARE construct [1].
- `expr equalop x` is an allowed condition in ATOMIC UPDATE COMPARE. [1]


### Data-sharing Clauses and Directives
- Using `COMMON` block variables in an `EQUIVALENCE` statement in `THREADPRIVATE` directives.

[1] Code generation for ATOMIC UPDATE COMPARE is not implemented yet.
