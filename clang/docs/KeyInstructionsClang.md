# Key Instructions in Clang

Key Instructions is an LLVM feature that reduces the jumpiness of optimized code debug stepping by discriminating the significance of instrucions that make up source language statements. This document explains how Clang applies the necessary metadata.

## Implementation

See the [LLVM docs](../../llvm/docs/KeyInstructionsDebugInfo.md) for general info about the feature (and LLVM implementation details).

Clang needs to annotate key instructions with the new metadata. Variable assignments (stores, memory intrinsics), control flow (branches and their conditions, some unconditional branches), and exception handling instructions are annotated. Calls are ignored as they're unconditionally marked `is_stmt`. This is achieved with a few simple constructs:

Class `ApplyAtomGroup` - This is a scoped helper similar to `ApplyDebugLocation` that creates a new source atom group which instructions can be added to. It's used during CodeGen to declare that a new source atom has started, e.g. in `CodeGenFunction::EmitBinaryOperatorLValue`.

`CodeGenFunction::addInstToCurrentSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup)` adds an instruction (and a backup instruction if non-null) to the current "atom group" defined with `ApplyAtomGroup`. The Key Instruction gets rank 1, and backup instructions get higher ranks (the function looks through casts, applying increasing rank as it goes). There are a lot of sites in Clang that need to call this (mostly stores and store-like instructions). Most stores created through `CGBuilderTy` are annotated, but some that don't need to be key are not. It's important to remember that if there's no active atom group, i.e. no active `ApplyAtomGroup` instance, then `addInstToCurrentSourceAtom` does not annotate the instructions.

`CodeGenFunction::addInstToNewSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup)` adds an instruction (and a backup instruction if non-null) to a new "atom group". Currently mostly used in loop handling code.

`CodeGenFunction::addInstToSpecificSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup, uint64_t Atom)` adds the instruction (and backup instruction if non-null) to the specific group `Atom`. This is currently only used for `rets` which is explored in the examples below. Special handling is needed due to the fact that an existing atom group needs to be reused in some circumstances, so neither of the other helper functions are appropriate.

## Examples

A simple example walk through:
```
void fun(int a) {
  int b = a;
}
```

There are two key instructions here, the assignment and the implicit return. We want to emit metadata that looks like this:

```
define hidden void @_Z3funi(i32 noundef %a) #0 !dbg !11 {
entry:
  %a.addr = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr %a.addr, align 4, !dbg !DILocation(line: 2, scope: !11, atomGroup: 1, atomRank: 2)
  store i32 %0, ptr %b, align 4,       !dbg !DILocation(line: 2, scope: !11, atomGroup: 1, atomRank: 1)
  ret void,                            !dbg !DILocation(line: 3, scope: !11, atomGroup: 2, atomRank: 1)
}
```

The store is the key instruction for the assignment (`atomGroup` 1). The instruction corresponding to the final (and in this case only) RHS value, the load from `%a.addr`, is a good backup location for `is_stmt` if the store gets optimized away. It's part of the same source atom, but has lower `is_stmt` precedence, so it gets a higher `atomRank`. This is achieved by starting an atom group with `ApplyAtomGroup` for the source atom (in this case a variable init) in `EmitAutoVarInit`. The instructions (both key and backup) are then annotated by call to `addInstToCurrentSourceAtom` called from `EmitStoreOfScalar`.

The implicit return is also key (`atomGroup` 2) so that it's stepped on, to match existing non-key-instructions behaviour. This is achieved by calling  `addInstToNewSourceAtom` from within `EmitFunctionEpilog`.

Explicit return statements are handled uniquely. Rather than emit a `ret` for each `return` Clang, in all but the simplest cases (as in the first example) emits a branch to a dedicated block with a single `ret`. That branch is the key instruction for the return statement. If there's only one branch to that block, because there's only one `return` (as in this example), Clang folds the block into its only predecessor. Handily `EmitReturnBlock` returns the `DebugLoc` associated with the single branch in that case, which is fed into `addInstToSpecificSourceAtom` to ensure the `ret` gets the right group.
