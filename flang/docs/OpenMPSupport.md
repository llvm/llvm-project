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
messages, which are considered authoritative. Partial support for OpenMP 4.0 is also available and currently under active development.
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
| simd construct                                             | P      | linear clause is not supported |
| declare simd construct                                     | N      | |
| do simd construct                                          | P      | linear clause is not supported |
| target data construct                                      | P      | device clause not supported |
| target construct                                           | P      | device clause not supported |
| target update construct                                    | P      | device clause not supported |
| declare target directive                                   | P      | |
| teams construct                                            | P      | reduction clause not supported |
| distribute construct                                       | P      | dist_schedule clause not supported |
| distribute simd construct                                  | P      | dist_schedule and linear clauses are not supported |
| distribute parallel loop construct                         | P      | dist_schedule clause not supported |
| distribute parallel loop simd construct                    | P      | dist_schedule and linear clauses are not supported |
| depend clause                                              | Y      | |
| declare reduction construct                                | N      | |
| atomic construct extensions                                | Y      | |
| cancel construct                                           | Y      | |
| cancellation point construct                               | Y      | |
| parallel do simd construct                                 | P      | linear clause is not supported |
| target teams construct                                     | P      | device and reduction clauses are not supported |
| teams distribute construct                                 | P      | reduction and dist_schedule clauses not supported |
| teams distribute simd construct                            | P      | reduction, dist_schedule and linear clauses are not supported |
| target teams distribute construct                          | P      | device, reduction and dist_schedule clauses are not supported |
| teams distribute parallel loop construct                   | P      | reduction and dist_schedule clauses are not supported |
| target teams distribute parallel loop construct            | P      | device, reduction and dist_schedule clauses are not supported |
| teams distribute parallel loop simd construct              | P      | reduction, dist_schedule, and linear clauses are not supported |
| target teams distribute parallel loop simd construct       | P      | device, reduction, dist_schedule and linear clauses are not supported |
