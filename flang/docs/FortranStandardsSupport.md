<!--===- docs/FortranStandardsSupport.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Flang Fortran Standards Support

```{contents}
---
local:
---
```

This document summarizes Fortran standards support in Flang. The information is only provided as a guideline. The
TODOs/Not Yet Implemented messages emitted by the compiler for unimplemented features should be treated as authoritative.
Standards support is provided upto Fortran 2008 for now. It will be extended later for Fortran 2018 and Fortran 2023.

The standards support information is provided as a table with three columns that are self explanatory. The Status column uses
the letters **P**, **Y**, **N** for the implementation status:
- **P** : When the implementation is incomplete for a few cases
- **Y** : When the implementation is complete
- **N** : When the implementation is absent

Note 1 : No distinction is made between the support in the Parser/Semantics and MLIR or Lowering support.
Note 2 : Besides the features listed below a few intrinsics like MIN/MAX are not supported for a few cases with CHARACTER type.

## Fortran 2023
TBD

## Fortran 2018
TBD

## Fortran 2008
All features except those listed in the following table are supported.

| Feature                                                    | Status | Comments                                                |
|------------------------------------------------------------|--------|---------------------------------------------------------|
| Coarrays                                                   | N      | Lowering and runtime support is not implemented         |
| do concurrent                                              | P      | Sequential execution works. Parallel support in progress|
| Internal procedure as an actual argument or pointer target | Y      | Current implementation requires stack to be executable. See [Proposal](InternalProcedureTrampolines.md) |

## Fortran 2003
All features except those listed in the following table are supported.

| Feature                                                    | Status | Comments                                                |
|------------------------------------------------------------|--------|---------------------------------------------------------|
| Parameterized Derived Types                                | P      | PDT with length type parameters is not supported. See [Proposal](ParameterizedDerivedTypes.md) |
| Assignment to allocatable                                  | P      | Assignment to whole allocatable in FORALL is not implemented       |
| The VOLATILE attribute                                     | P      | VOLATILE in procedure interfaces is not implemented     |
| Asynchronous input/output                                  | P      | IO will happen synchronously                            |

## FORTRAN 77
All features are supported.

## Fortran 90
All features are supported.

## Fortran 95
All features are supported.
