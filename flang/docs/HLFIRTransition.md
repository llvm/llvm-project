# Transition of Lowering to HLFIR

This section was extracted from [HighLevelFIR.md](High-Level Fortran IR (HLFIR)). 
This information is no longer relevant to the current state of HLFIR
lowering, but could be useful as a historical reference.

## Transition Plan

The new higher-level steps proposed in this document will require significant
refactoring of lowering. Codegen should not be impacted since the current FIR
will remain untouched.

A lot of the code in lowering generating Fortran features (like an intrinsic or
how to do assignments) is based on the fir::ExtendedValue concept. This
currently is a collection of mlir::Value that allows describing a Fortran object
(either a variable or an evaluated expression result). The variable and
expression concepts described above should allow to keep an interface very
similar to the fir::ExtendedValue, but having the fir::ExtendedValue wrap a
single value or mlir::Operation* from which all of the object entity
information can be inferred.

That way, all the helpers currently generating FIR from fir::ExtendedValue could
be kept and used with the new variable and expression concepts with as little
modification as possible.

The proposed plan is to:
- 1. Introduce the new HLFIR operations.
- 2. Refactor fir::ExtendedValue so that it can work with the new variable and
     expression concepts (requires part of 1.).
- 3. Introduce the new translation passes, using the fir::ExtendedValue helpers
     (requires 1.).
- 3.b Introduce the new optimization passes (requires 1.).
- 4. Introduce the fir.declare and hlfir.finalize usage in lowering (requires 1.
     and 2. and part of 3.).

The following steps might have to be done in parallel of the current lowering,
to avoid disturbing the work on performance until the new lowering is complete
and on par.

- 5. Introduce hlfir.designate and hlfir.associate usage in lowering.
- 6. Introduce lowering to hlfir.assign (with RHS that is not a hlfir.expr),
     hlfir.ptr_assign.
- 7. Introduce lowering to hlfir.expr and related operations.
- 8. Introduce lowering to hlfir.forall.

At that point, lowering using the high-level FIR should be in place, allowing
extensive testing.
- 9. Debugging correctness.
- 10. Debugging execution performance.

The plan is to do these steps incrementally upstream, but for lowering this will
most likely be safer to do have the new expression lowering implemented in
parallel upstream, and to add an option to use the new lowering rather than to
directly modify the current expression lowering and have it step by step
equivalent functionally and performance wise.
