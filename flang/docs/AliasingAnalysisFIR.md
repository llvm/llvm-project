<!--===- docs/Aliasing.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Aliasing analysis in FIR

This document describes the design of Alias Analysis for the FIR dialect, using the MLIR infrastructure. The intention is to use this analysis as a building block for more advanced analyses such as global code motion. 

The result will be 
1. A class, implementing the  [AliasAnalysis](https://github.com/llvm/llvm-project/blob/189900eb149bb55ae3787346f57c1ccbdc50fb3c/mlir/include/mlir/Analysis/AliasAnalysis.h#L232) interface. It will be able to answer two types of queries:
   - AliasResult Alias (Value lhs, Value rhs)
    
     Given two memory references, return their aliasing behavior

    - ModRefResult getModRef(Operation *op, Value location)
        
      The possible results of whether a memory access modifies or references a memory location. This will not be performing a dataflow analysis. It will not take into account the instructions in the paths between the arguments. It will merely factor the type of side effects into the aliasing results. 

1. A testing pass, performing unit testing on the analysis.
   
   The pass will take FIR as an input, look for some predefined annotations  and report their aliasing behavior. This will provide a development framework that will allow for the initial implementation of a coarse analysis that can then be refined on an ongoing basis.


## Definitions
### Memory side effect or side effect:
1. Presence of MemoryEffectOpInterface

   A side effect will be determined by the MemoryEffectOpInterface. The interface can inform for each operand of an MLIR operation, whether there is a side effect on it or not. The possible side effects are:

    - Allocate
    - Free
    - Read
    - Write
     
   An atomic read-modify-write can have both a read and write side-effect on its operands
   
   For the implementation of getModRef,  the effects will also be classified as 
   - Modifying (Allocate, Free, Write)
   - Non-modifying (Read)
  

1. Absence of MemoryEffectOpInterface

   In the absence of a MemoryEffectOpInterface, the conservative assumption will have to be that there is a modifying effect on all operands. 

   As far as user calls are concerned, this rule will be relaxed in the presence of the `INTENT(IN)` or `VALUE` attribute.

   Runtime calls are not covered by the Fortran language. They are C calls which can take raw pointers by value. We will need to define some convention for their aliasing behavior


### Memory reference: 
Any SSA value on the RHS of an operation with a memory side effect as defined above.

### Memory source:
The storage from which a memory reference is sourced. A memory reference may not be the source of the storage and may be reached by following the use-def chain through specific operations such as fir.convert, fir.coordinate_of, fir.array_coor, fir.embox, fir.rebox, fir.box_addr…. 

Possible sources are:
- The LHS of an operation with Allocate side effect, this could be local or heap
- A global symbol: the RHS of fir.address_of (Note that a global symbol is not an SSA value but an attribute)
- A dummy argument: a block argument of the entry block of a func.func operation.
- Indirect source: load of memory reference stored at another memory reference
- Unknown source: when the use-def traversal does not reach any of the sources above. 

### Path to source:
Additional information can be collected on the way to the source such as type (fir.heap, fir.ptr), attributes (fir.target) and use-def chains (fir.coordinate_of, fir.array_coor, fir.declare...). Constant paths can help disambiguate aliasing.


Because of block arguments, a memory reference may have multiple sources. If a block argument is encountered, all predecessors will have to be visited. When querying the aliasing behavior of two memory references the cartesian product of all paths need to be considered.

### Pointer type
A type `fir.box<fir.ptr<T>>` or `fir.ptr<T>`

## Aliasing rules
The goal is to match [Fortran’s rule for aliasing](Aliasing.md). However FIR is all we have at this stage so the hope is that we can define an algorithm using the information from FIR to properly model Fortran’s aliasing rules. Wherever there is a gap, we may have to refine the algorithm, add information in FIR or both. Though, with the introduction of the fir.declare operation, most of the source level information relevant to aliasing will be populated in FIR.

The first attempt to determine aliasing will be at the coarsest level: the source level. The answer to the query will be ‘yes’, ‘no’, ‘maybe’. If the answer is ‘yes’ or ‘no’, the query is complete. If the answer is ‘maybe’ then further analysis is required until a definite answer is reached. If no finer analysis is available then 'maybe' is returned.

### Coarse rules
Distinct sources are assumed to not alias except in the following cases:
1. A pointer type source may alias with any other pointer type source.
1. A source with the fir.target attribute may alias with any other pointer type source.
1. Indirect sources of non pointer type and unknown sources may alias with any source.

