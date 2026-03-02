<!--===- docs/OpenMPSupport.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Flang OpenMP Support

```{contents}
---
local:
---
```

This document outlines the OpenMP API features supported by Flang. It is intended as a general reference.
For the most accurate information on unimplemented features, rely on the compiler’s TODO or “Not Yet Implemented”
messages, which are considered authoritative. Flang provides complete implementation of the OpenMP 3.1 specification and partial implementation of OpenMP 4.0, with continued development efforts aimed at extending full support for the latter.
The table below outlines the current status of OpenMP 4.0 feature support.
Work is ongoing to add support for OpenMP 4.5 and newer versions; a support statement for these will be shared in the future.
The table entries are derived from the information provided in the Version Differences subsection of the Features History section in the OpenMP standard.

The feature support information is provided as a table with three columns that are self explanatory. The Status column uses
the letters **P**, **Y**, **N** for the implementation status:
- **P** : Partial. When the implementation is incomplete for a few cases
- **Y** : Yes. When the implementation is complete
- **N** : No. When the implementation is absent

Note : No distinction is made between the support in Parser/Semantics, MLIR, Lowering or the OpenMPIRBuilder.

## OpenMP 4.0

| Feature                                                    | Status | Comments                                                |
|------------------------------------------------------------|--------|---------------------------------------------------------|
| proc_bind clause                                           | Y      | |
| simd construct                                             | P      | Implicit linearization is skipped if iv is a pointer or allocatable|
| declare simd construct                                     | N      | |
| do simd construct                                          | P      | Implicit linearization is skipped if iv is a pointer or allocatable |
| target data construct                                      | Y      | |
| target construct                                           | Y      | |
| target update construct                                    | Y      | |
| declare target directive                                   | Y      | |
| teams construct                                            | Y      | |
| distribute construct                                       | Y      | |
| distribute simd construct                                  | P      | Implicit linearization is skipped if iv is a pointer or allocatable |
| distribute parallel loop construct                         | Y      | |
| distribute parallel loop simd construct                    | P      | Implicit linearization is skipped if iv is a pointer or allocatable |
| depend clause                                              | Y      | |
| declare reduction construct                                | N      | |
| atomic construct extensions                                | Y      | |
| cancel construct                                           | Y      | |
| cancellation point construct                               | Y      | |
| parallel do simd construct                                 | P      | Implicit linearization is skipped if iv is a pointer or allocatable |
| target teams construct                                     | Y      | |
| teams distribute construct                                 | Y      | |
| teams distribute simd construct                            | P      | Implicit linearization is skipped if iv is a pointer or allocatable |
| target teams distribute construct                          | Y      | |
| teams distribute parallel loop construct                   | Y      | |
| target teams distribute parallel loop construct            | Y      | |
| teams distribute parallel loop simd construct              | P      | Implicit linearization is skipped if iv is a pointer or allocatable |
| target teams distribute parallel loop simd construct       | P      | Implicit linearization is completely skipped |

## Extensions
### ATOMIC construct
The implementation of the ATOMIC construct follows OpenMP 6.0 with the following extensions:
- `x = x` is an allowed form of ATOMIC UPDATE.
This is motivated by the fact that the equivalent forms `x = x+0` or `x = x*1` are allowed.
- Explicit type conversions are allowed in ATOMIC READ, WRITE or UPDATE constructs, and in the capture statement in ATOMIC UPDATE CAPTURE.
The OpenMP spec requires intrinsic- or pointer-assignments, which include (as per the Fortran standard) implicit type conversions.  Since such conversions need to be handled, allowing explicit conversions comes at no extra cost.
- A literal `.true.` or `.false.` is an allowed condition in ATOMIC UPDATE COMPARE. [1]
- A logical variable is an allowed form of the condition even if its value is not computed within the ATOMIC UPDATE COMPARE construct [1].
- `expr equalop x` is an allowed condition in ATOMIC UPDATE COMPARE. [1]

[1] Code generation for ATOMIC UPDATE COMPARE is not implemented yet.
