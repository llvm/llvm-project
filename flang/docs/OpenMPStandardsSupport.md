<!--===- docs/FortranStandardsSupport.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Flang OpenMP Standards Support

```{contents}
---
local:
---
```

This document summarizes OpenMP standards support in Flang. The information is only provided as a guideline. The
TODOs/Not Yet Implemented messages emitted by the compiler for unimplemented features should be treated as authoritative.
Standards support is provided upto OpenMP 4.0 for now. It will be extended later for OpenMP 4.5, OpenMP 5.* and OpenMP 6.0.

The standards support information is provided as a table with three columns that are self explanatory. The Status column uses
the letters **P**, **Y**, **N** for the implementation status:
- **P** : When the implementation is incomplete for a few cases
- **Y** : When the implementation is complete
- **N** : When the implementation is absent

Note : No distinction is made between the support in the Parser/Semantics, MLIR or Lowering support, and OpenMPIRBuilder support.

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
All features except a few corner cases in atomic, copyin constructs/clauses are supported 
