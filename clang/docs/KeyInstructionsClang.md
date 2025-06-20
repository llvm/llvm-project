# Key Instructions in Clang

Key Instructions is an LLVM feature that reduces the jumpiness of optimized code debug stepping. This document explains how Clang applies the necessary metadata.

## Implementation

See the [LLVM docs](../../llvm/docs/KeyInstructionsDebugInfo.md) for general info about the feature (and LLVM implementation details).

Clang needs to annotate key instructions with the new metadata. Variable assignments (stores, memory intrinsics), control flow (branches and their conditions, some unconditional branches), and exception handling instructions are annotated. Calls are ignored as they're unconditionally marked `is_stmt`. This is achieved with a few simple constructs:

Class `ApplyAtomGroup` - This is a scoped helper similar to `ApplyDebugLocation` that creates a new source atom group which instructions can be added to. It's used during CodeGen to declare that a new source atom has started, e.g. in `CodeGenFunction::EmitBinaryOperatorLValue`.

`CodeGenFunction::addInstToCurrentSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup)` adds an instruction (and a backup instruction if non-null) to the current "atom group" defined with `ApplyAtomGroup`. The Key Instruction gets rank 1, and backup instructions get higher ranks (the function looks through casts, applying increasing rank as it goes). There are a lot of sites in Clang that need to call this (mostly stores and store-like instructions). FIXME?: Currently it's called at the CGBuilderTy callsites; it could instead make sense to always call the function inside the CGBuilderTy calls, with some calls opting out.

`CodeGenFunction::addInstToNewSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup)` adds an instruction (and a backup instruction if non-null) to a new "atom group". Currently mostly used in loop handling code.

There are a couple of other helpers, including `addRetToOverrideOrNewSourceAtom` used for `rets` which is covered in the examples below.

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

The store is the key instruction for the assignment (`atomGroup` 1). The instruction corresponding to the final (and in this case only) RHS value, the load from `%a.addr`, is a good backup location for `is_stmt` if the store gets optimized away. It's part of the same source atom, but has lower `is_stmt` precedence, so it gets a higher `atomRank`.

This is all handled during CodeGen. The atom group is set here:
```
>  clang::CodeGen::ApplyAtomGroup::ApplyAtomGroup(clang::CodeGen::CGDebugInfo * DI) Line 187
   clang::CodeGen::CodeGenFunction::EmitAutoVarInit(const clang::CodeGen::CodeGenFunction::AutoVarEmission & emission) Line 1961
   clang::CodeGen::CodeGenFunction::EmitAutoVarDecl(const clang::VarDecl & D) Line 1361
   clang::CodeGen::CodeGenFunction::EmitVarDecl(const clang::VarDecl & D) Line 219
   clang::CodeGen::CodeGenFunction::EmitDecl(const clang::Decl & D) Line 164
   clang::CodeGen::CodeGenFunction::EmitDeclStmt(const clang::DeclStmt & S) Line 1611
   clang::CodeGen::CodeGenFunction::EmitSimpleStmt(const clang::Stmt * S, llvm::ArrayRef<clang::Attr const *> Attrs) Line 466
   clang::CodeGen::CodeGenFunction::EmitStmt(const clang::Stmt * S, llvm::ArrayRef<clang::Attr const *> Attrs) Line 72
   clang::CodeGen::CodeGenFunction::EmitCompoundStmtWithoutScope(const clang::CompoundStmt & S, bool GetLast, clang::CodeGen::AggValueSlot AggSlot) Line 556+
   clang::CodeGen::CodeGenFunction::EmitFunctionBody(const clang::Stmt * Body) Line 1307
```

And the DILocations are updated here:
```
>  clang::CodeGen::CodeGenFunction::addInstToCurrentSourceAtom(llvm::Instruction * KeyInstruction, llvm::Value * Backup, unsigned char KeyInstRank) Line 2551
   clang::CodeGen::CodeGenFunction::EmitStoreOfScalar(llvm::Value * Value, clang::CodeGen::Address Addr, bool Volatile, clang::QualType Ty, clang::CodeGen::LValueBaseInfo BaseInfo, clang::CodeGen::TBAAAccessInfo TBAAInfo, bool isInit, bool isNontemporal) Line 2133
   clang::CodeGen::CodeGenFunction::EmitStoreOfScalar(llvm::Value * value, clang::CodeGen::LValue lvalue, bool isInit) Line 2152
   clang::CodeGen::CodeGenFunction::EmitStoreThroughLValue(clang::CodeGen::RValue Src, clang::CodeGen::LValue Dst, bool isInit) Line 2478
   clang::CodeGen::CodeGenFunction::EmitScalarInit(const clang::Expr * init, const clang::ValueDecl * D, clang::CodeGen::LValue lvalue, bool capturedByInit) Line 805
   clang::CodeGen::CodeGenFunction::EmitExprAsInit(const clang::Expr * init, const clang::ValueDecl * D, clang::CodeGen::LValue lvalue, bool capturedByInit) Line 2088
   clang::CodeGen::CodeGenFunction::EmitAutoVarInit(const clang::CodeGen::CodeGenFunction::AutoVarEmission & emission) Line 2050
   clang::CodeGen::CodeGenFunction::EmitAutoVarDecl(const clang::VarDecl & D) Line 1361
   clang::CodeGen::CodeGenFunction::EmitVarDecl(const clang::VarDecl & D) Line 219
   clang::CodeGen::CodeGenFunction::EmitDecl(const clang::Decl & D) Line 164
   clang::CodeGen::CodeGenFunction::EmitDeclStmt(const clang::DeclStmt & S) Line 1611
   clang::CodeGen::CodeGenFunction::EmitSimpleStmt(const clang::Stmt * S, llvm::ArrayRef<clang::Attr const *> Attrs) Line 466
   clang::CodeGen::CodeGenFunction::EmitStmt(const clang::Stmt * S, llvm::ArrayRef<clang::Attr const *> Attrs) Line 72
   clang::CodeGen::CodeGenFunction::EmitCompoundStmtWithoutScope(const clang::CompoundStmt & S, bool GetLast, clang::CodeGen::AggValueSlot AggSlot) Line 556
   clang::CodeGen::CodeGenFunction::EmitFunctionBody(const clang::Stmt * Body) Line 1307

```

The implicit return is also key (`atomGroup` 2) so that it's stepped on, to match existing non-key-instructions behaviour.

```
>  clang::CodeGen::CodeGenFunction::addRetToOverrideOrNewSourceAtom(llvm::ReturnInst * Ret, llvm::Value * Backup, unsigned char KeyInstRank) Line 2567
   clang::CodeGen::CodeGenFunction::EmitFunctionEpilog(const clang::CodeGen::CGFunctionInfo & FI, bool EmitRetDbgLoc, clang::SourceLocation EndLoc) Line 3839
   clang::CodeGen::CodeGenFunction::FinishFunction(clang::SourceLocation EndLoc) Line 433
```

`addRetToOverrideOrNewSourceAtom` is a special function used for handling `ret`s. In this case it simply replaces the DILocation with the `atomGroup` and `atomRank` set, adding it to its own atom.

To demonstrate why `ret`s need special handling, we need to look at a more "complex" example, below.

```
int fun(int a) {
  return a;
}
```

Rather than emit a `ret` for each `return` Clang, in all but the simplest cases (as in the first example) emits a branch to a dedicated block with a single `ret`. That branch is the key instruction for the return statement. If there's only one branch to that block, because there's only one `return` (as in this example), Clang folds the block into its only predecessor. We need to do some accounting to transfer the `atomGroup` number to the `ret` when that happens:

When we hit the special-casing code that knows we've only got one block (the IR looks like this):
```
entry:
  %a.addr = alloca i32, align 4
  %allocapt = bitcast i32 undef to i32
  store i32 %a, ptr %a.addr, align 4
  br label %return, !dbg !6
```

...remember the branch-to-return-block's `atomGroup`:

```
>  clang::CodeGen::CGDebugInfo::setRetInstSourceAtomOverride(unsigned __int64 Group) Line 168
   clang::CodeGen::CodeGenFunction::EmitReturnBlock() Line 332
   clang::CodeGen::CodeGenFunction::FinishFunction(clang::SourceLocation EndLoc) Line 415
```

And apply it to the `ret` when it's added:
```
>  clang::CodeGen::CodeGenFunction::addRetToOverrideOrNewSourceAtom(llvm::ReturnInst * Ret, llvm::Value * Backup, unsigned char KeyInstRank) Line 2567
   clang::CodeGen::CodeGenFunction::EmitFunctionEpilog(const clang::CodeGen::CGFunctionInfo & FI, bool EmitRetDbgLoc, clang::SourceLocation EndLoc) Line 3839
   clang::CodeGen::CodeGenFunction::FinishFunction(clang::SourceLocation EndLoc) Line 433
```
