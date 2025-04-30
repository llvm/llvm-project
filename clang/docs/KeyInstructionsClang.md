# Key Instructions in Clang

Key Instructions reduces the jumpiness of optimized code debug stepping. This document explains the feature and how it is implemented in LLVM. For Clang support please see the Clang docs.

## Status

In development - some details may change with little notice.

Tell Clang [not] to produce Key Instructions metadata with `-g[no-]key-instructions`. This also sets the LLVM flag `-dwarf-use-key-instructions`, so it interprets Key Instructions metadata when producing the DWARF line table.

The feature improves optimized code stepping; it's intended for the feature to be used with optimisations enabled. Although the feature works at O0 it is not recommended because in some cases the effect of editing variables may not always be immediately realised. (This is a quirk of the current implementation, rather than fundemental limitation, covered in more detail later).

There is currently no plan to support CodeView.

## Implementation

See the [LLVM docs](../../llvm/docs/KeyInstructionsDebugInfo.md) for general info about the feature (and LLVM implementation details).

Clang needs to annotate key instructions with the new metadata. Variable assignments (stores, memory intrinsics), control flow (branches and their conditions, some unconditional branches), and exception handling instructions are annotated. Calls are ignored as they're unconditionally marked `is_stmt`. This is achieved with a few simple constructs:

Class `ApplyAtomGroup` - This is a scoped helper similar to `ApplyDebugLocation` that creates a new source atom group which instructions can be added to. It's used during CodeGen to declare that a new source atom has started, e.g. in `CodeGenFunction::EmitBinaryOperatorLValue`.

`CodeGenFunction::addInstToCurrentSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup)` adds an instruction (and a backup instruction if non-null) to the current "atom group" defined with `ApplyAtomGroup`. The Key Instruction gets rank 1, and backup instructions get higher ranks (the function looks through casts, applying increasing rank as it goes). There are a lot of sites in Clang that need to call this (mostly stores and store-like instructions). FIXME?: Currently it's called at the CGBuilderTy callsites; it could instead make sense to always call the function inside the CGBuilderTy calls, with some calls opting out.

`CodeGenFunction::addInstToNewSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup)` adds an instruction (and a backup instruction if non-null) to a new "atom group". Currently mostly used in loop handling code.
