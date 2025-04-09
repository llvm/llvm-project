<!--===- docs/FortranStandardsSupport.md

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
messages, which are considered authoritative.  With the exception of a few corner cases, Flang
offers full support for OpenMP 3.1 ([See details here](#OpenMP 3.1, OpenMP 2.5, OpenMP 1.1)).
Partial support for OpenMP 4.0 is also available and currently under active development. The table below outlines the current status of OpenMP 4.0 feature support.
The table below details the current support for OpenMP 4.0 features. Work is ongoing to add support
for OpenMP 4.5 and newer versions; an official support statement for these will be shared in the future.

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
| simd construct                                             | P      | Some clauses are not supported |
| declare simd construct                                     | N      | |
| do simd construct                                          | Y      | |
| target data construct                                      | P      | |
| target construct                                           | P      | |
| target update construct                                    | P      | |
| declare target directive                                   | P      | |
| teams construct                                            | P      | |
| distribute construct                                       | P      | |
| distribute simd construct                                  | P      | |
| distribute parallel loop construct                         | P      | |
| distribute parallel loop simd construct                    | P      | |
| depend clause                                              | P      | Depend clause with array sections are not supported |
| declare reduction construct                                | N      | |
| atomic construct extensions                                | Y      | |
| cancel construct                                           | N      | |
| cancellation point construct                               | N      | |
| parallel do simd construct                                 | Y      | |
| target teams construct                                     | P      | |
| teams distribute construct                                 | P      | |
| teams distribute simd construct                            | P      | |
| target teams distribute construct                          | P      | |
| teams distribute parallel loop construct                   | P      | |
| target teams distribute parallel loop construct            | P      | |
| teams distribute parallel loop simd construct              | P      | |
| target teams distribute parallel loop simd construct       | P      | |

## OpenMP 3.1, OpenMP 2.5, OpenMP 1.1
All features except a few corner cases in atomic (complex type, different but compatible types in lhs and rhs), threadprivate (character type) constructs/clauses are supported.
