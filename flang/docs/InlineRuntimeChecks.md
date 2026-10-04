<!--===- docs/InlineRuntimeChecks.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Inline runtime checks in Flang with `fir.assert`

Status: draft design, for discussion.
Context: [llvm/llvm-project#223287](https://github.com/llvm/llvm-project/pull/223287).
Source references are to `llvm-project` at `fe1de64cb3c2` (2026-10-02). Line
numbers drift, so each reference also names the function or pattern.

Abbreviations used in the tables:

| Short | Path |
|---|---|
| `SHI` | `flang/lib/Optimizer/HLFIR/Transforms/SimplifyHLFIRIntrinsics.cpp` |
| `OB` | `flang/lib/Optimizer/HLFIR/Transforms/OptimizedBufferization.cpp` |
| `IHA` | `flang/lib/Optimizer/HLFIR/Transforms/InlineHLFIRAssign.cpp` |
| `SAA` | `flang/lib/Optimizer/HLFIR/Transforms/SeparateAllocatableAssign.cpp` |
| `C2F` | `flang/lib/Optimizer/HLFIR/Transforms/ConvertToFIR.cpp` |
| `SI` | `flang/lib/Optimizer/Transforms/SimplifyIntrinsics.cpp` (FIR level) |
| `IC` | `flang/lib/Optimizer/Builder/IntrinsicCall.cpp` |
| `X2H` | `flang/lib/Lower/ConvertExprToHLFIR.cpp` |
| `RT/` | `flang-rt/lib/runtime/` |

## 1. Problem

Flang relies on the Fortran runtime for almost all dynamic checks: shape
conformance, allocation status, argument consistency of intrinsics. Every
time an HLFIR or FIR pass replaces a runtime call with inline code, the check
silently disappears. Three consequences:

1. **Behavior depends on the optimization level.** The inlining passes mostly
   run at `-O1` and above, so the same non-conforming program stops with a
   clear message at `-O0` and does something else at `-O2`.
2. **The inline code is often memory-unsafe on non-conforming input, not just
   wrong.** Loop bounds are taken from one operand and used to index another
   (`fir::factory::deduceOptimalExtents`). A size mismatch becomes an
   out-of-bounds read or write instead of an error.
3. **Some checks the standard requires are missing on every path.** For
   example, the inline ALLOCATE/DEALLOCATE of intrinsic-type allocatables
   does not check the allocation status (F2018 9.7.1.3, 9.7.3.2, 9.7.4).

The few inline checks that exist today (MOD/MODULO `P==0`, NEAREST `S==0`,
STORAGE_SIZE, the realloc-with-scalar-RHS case) use `fir.if` plus
`fir.call @_FortranAReportFatalUserError`. That pattern:

- has no data dependence on the operation it protects, so passes may move the
  protected operation above the check (MLIR models neither "may not return"
  nor non-local control flow: `mlir/docs/Rationale/SideEffectsAndSpeculation.md`);
- introduces a call with unknown memory effects, which blocks alias-based
  optimizations;
- does not tell LLVM the call never returns (the FIR declaration has no
  `noreturn`), so LLVM cannot use the checked condition afterwards.

## 2. Goals and non-goals

Goals:

- Keep every user-facing check the runtime performs when the call is inlined,
  and report it with the runtime's message format:
  `fatal Fortran runtime error(<file>:<line>): <message>`.
- Keep the checks cheap and transparent to optimization. They must not block
  store-to-load forwarding, CSE of loads, LICM of loads, or alias analysis.
- Let checks be hoisted, merged, folded, and dropped under an option.
- Provide one mechanism for future opt-in checks (`-fcheck=bounds`,
  `pointer`, `do`, `bits`).

Non-goals:

- Detecting undefined POINTER association status (only disassociated, that
  is null, is detectable).
- Checking explicit-shape dummy size against the actual argument in general
  (the callee only receives an address).
- Changing the runtime's own checks, apart from the bugs listed in the
  appendix.

## 3. The `fir.assert` operation

### 3.1 Semantics and syntax

```
%g:N = fir.assert %ok, <kind>, "<message>" [values(%v0, ... : T0, ...)] [guard(%x0, ... : U0, ...)]
```

- `%ok` (`i1`) is the condition that must hold.
- `<kind>` is a check category (section 4). Options act on categories.
- `<message>` is the error text, a `printf` format.
- `values(%v0, ... : T0, ...)` are optional integer or `index` SSA values
  (`T0`, ... are their types) printed into the message, one per `%jd`
  directive. They are only read on the failure path. For example, the two
  extents in `"DOT_PRODUCT: SIZE(VECTOR_A) is %jd but SIZE(VECTOR_B) is %jd"`.
- `guard(%x0, ... : U0, ...)` are the optional guarded values, of any type.
  The op has one result per guarded value, with the same type (`%g#i` has
  type `Ui`).
- If `%ok` is true, each result equals the corresponding guarded value and
  nothing else happens. Otherwise the program terminates through the Fortran
  runtime with the formatted message.
- Operations that must not run before the check consume the guarded results.
  The ordering is then enforced by SSA dominance, for every pass, without any
  pass having to know about `fir.assert`. Asserts without a guard only order
  themselves with respect to I/O and calls (section 3.2).

What to guard: the value from which the dangerous access is derived. For
inlined loops this is the **extent used as the loop bound** (or the
`fir.shape` operands): every access in the loop depends on the induction
variable, which depends on the guarded extent. Guarding extents rather than
array entities avoids threading `!hlfir.expr` values through the assert.

### 3.2 Traits

- `MemoryEffects<[MemWrite<FortranRuntimeResource>]>`, where
  `FortranRuntimeResource` is a non-addressable resource (like the existing
  `fir::DebuggingResource`).
  - The op is never trivially dead, so DCE keeps asserts without uses.
  - It stays ordered with calls and I/O, whose unknown effects cover every
    resource.
  - It does not touch user memory. FIR alias analysis skips non-addressable
    resources (`flang/lib/Optimizer/Analysis/AliasAnalysis.cpp`, `getModRef`),
    and MLIR CSE ignores writes to resources disjoint from a load's resource.
    So asserts do not block load CSE, store-to-load forwarding, or FIR LICM of
    loads. This also means `SimplifyHLFIRIntrinsics` can emit asserts in its
    first run (`allowNewSideEffects=false`, `SHI:3321-3333`): the concern
    there is new effects that stop CSE from merging loads, which this op does
    not have.
- Not speculatable.
- **Not transparent to other operations' folds and patterns.** The assert's
  own canonicalization removes it when `%ok` is a constant true (section 3.4).
  But no other rewrite may look through an assert to its guarded operand,
  because that drops the dependence. For example, in
  `%a = fir.box_addr %b; %a_ok = fir.assert ... guard(%a); %c = fir.embox %a_ok`,
  a fold that rewrites an `fir.embox` of a `fir.box_addr` into the original
  box must not replace `%c` with `%b`. In practice:
  - the op does not implement `ViewLikeOpInterface`;
  - it is not added to the helpers that skip `fir.declare` or `fir.convert`
    to find a defining op;
  - it is not given a folder that returns a guarded operand when `%ok` is
    unknown.

  Alias analysis may look through the assert, since it only answers
  aliasing queries and does not rewrite uses.

### 3.3 TableGen sketch

```
def fir_AssertOp : fir_Op<"assert", [AttrSizedOperandSegments,
    MemoryEffects<[MemWrite<FortranRuntimeResource>]>]> {
  let arguments = (ins I1:$condition,
                       fir_RuntimeCheckKindAttr:$kind,
                       StrAttr:$message,
                       Variadic<AnySignlessIntegerOrIndex>:$values,
                       Variadic<AnyType>:$guarded);
  let results = (outs Variadic<AnyType>:$results); // types == guarded types
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}
```

### 3.4 Canonicalization and redundancy elimination

- **Constant true:** erase the assert and replace its results with the guarded
  operands. This replaces the hand-written `fir::getIntIfConstant` early-outs
  of today's checks (`IC:6929-6931`). It must be a canonicalization pattern
  rather than only a `fold`: for an assert with no guard, an empty fold result
  means an in-place update, not an erasure.
- **Constant false:** keep it, with no diagnostic. A false condition is common
  in code that is unreachable but hard to prove unreachable at compile time
  (for example, after inlining or specialization on constant arguments), so a
  warning would be noisy.

  A later pass could erase the code strictly dominated by a constant-false
  assert. This is not proposed for the first version, and it must not run in
  any pipeline that may drop asserts (for example, device code with checks
  disabled). There, the assert would disappear together with the pruned
  code, and a program that should have stopped would silently produce
  wrong results.
- **Dominating assert with the same condition:** remove the later one, mapping
  its results to the earlier one's results. This is valid regardless of what
  lies between them and regardless of the message, because the earlier check
  fails first. Only remove it when the earlier assert guards everything the
  later one guards; otherwise the consumers would lose their dependence.
  Upstream CSE cannot do this: it skips ops with write effects
  (`mlir/lib/Transforms/Utils/CSE.cpp`, `simplifyOperation`). So this needs a
  small dominance-based pass (section 6).

### 3.5 Lowering

Lowering is done in `FIRToLLVMLowering`, where the IR is already a control-flow
graph:

```
// fir.assert %ok, conformance, "..." values(%a, %b : index, index) guard(%n : index)
llvm.cond_br %ok weights([2000, 1]), ^cont, ^fail
^fail:
  llvm.call @_FortranAReportFatalUserErrorValues(%msg, %file, %line, %a64, %b64, ...)
  llvm.unreachable
^cont:   // uses of the result are replaced with %n
```

- **Runtime entry.** `_FortranAReportFatalUserError(msg, source, line)`
  (`flang/include/flang/Runtime/stop.h`) is `[[noreturn]]` and device-callable
  (`RT_API_ATTRS`). It takes no values. Proposed addition:
  `ReportFatalUserErrorValues(msg, source, line, int64 v0..v3)` with a fixed
  number of `int64_t` arguments. That keeps the ABI simple on devices, and
  messages use `%jd` only. The message is passed to `printf`, so a literal
  `%` in a message must be escaped.
- **Declaration.** The runtime function declaration must carry `noreturn`,
  so LLVM can use the checked condition on the continuation path.
- **Source location.** Take it from the assert's own location. Resolve
  `FusedLoc` and `CallSiteLoc` to the innermost `FileLineColLoc`: today
  `fir::factory::locationToFilename` returns null for them, so inlined checks
  lose file and line.
- **Device code.** String globals must be created in the nearest symbol table
  (a `gpu.module` for device code), and the lowering must also exist in any
  device code generation pipeline. If a device runtime lacks the entry point,
  fall back to `printf` plus `llvm.trap`. The function is not in the OpenMP
  offload API group today, unlike `Abort`.

Lowering before `CFGConversion` (to `fir.if` plus call) is also possible, but
`fir.if` regions cannot end with `fir.unreachable`, and asserts would stop
being visible to the passes after that point.

### 3.6 Alternatives considered

**`fir.if` plus a runtime call (today's pattern).** Not retained, for the
reasons in section 1: there is no data dependence on the protected operation,
the call's unknown memory effects block alias-based optimizations, and LLVM
is not told that the call never returns.

**`cf.assert`.** Not retained. It writes the default resource, which is how
it models program termination, so it blocks store-to-load forwarding and
load CSE across it. It still does not order non-speculatable operations
without memory effects, such as `arith.remsi`, after it. Its LLVM lowering
prints with `puts` and calls `abort`, rather than reporting through the
Fortran runtime. It also has no guarded results.

**A result-less `fir.assert` (the current revision of
[#223287](https://github.com/llvm/llvm-project/pull/223287)).** The PR
defines `fir.assert` as a non-speculatable op with a
`MemWrite<FortranRuntimeResource>` effect, lowers it through `cf.assert`, and
teaches Flang LICM not to hoist non-speculatable operations across a
preceding assert in the same block. This design keeps that effect model
(section 3.2), which gives the right DCE, alias-analysis, and forwarding
behavior. It is not retained as is, because the ordering of the protected
operations is not expressed in the IR:

- MLIR has no notion of an operation that may not return. Every pass may
  assume that all operations in a block execute together, and may move a
  `fir.load` or an `arith.remsi` above an assert whose effects do not
  conflict with it. The LICM change protects one pass. Every other pass that
  moves or recreates operations would need the same fix, and future passes
  would have to remember it. Examples are sinking, scheduling, and rewrite
  patterns that build new operations at a different insertion point.
- Once a protected load or remainder is above the check, LLVM may infer from
  its undefined behavior that the check cannot fail, and delete it.
- The guarded results make the ordering a matter of SSA dominance, so no pass
  has to know about `fir.assert`. The lowering then goes directly to the
  Fortran runtime entry, so messages match the runtime's format (section
  3.5).

The design in this document is therefore the PR's op, with guarded results
and a direct lowering added.

**A region-based assert, like `shape.assuming`.** The protected operations go
inside a region attached to the assert, and the values used afterwards are
yielded out:

```
%r = fir.assert %ok, conformance, "..." -> (f32) {
  // inlined DOT_PRODUCT loop nest
  fir.result %sum : f32
}
```

Ordering is structural, so it is as safe as the guarded-result form. It is
not retained because the region gets in the way of the optimizations the
checks must not block:

- The protected code is usually a whole loop nest or the rest of a
  statement, so everything it defines has to be yielded through the region.
  An allocation-status check would wrap the rest of the block.
- Passes that work on a single block or match operation sequences see a
  region boundary instead. Examples are the `OptimizedBufferization` pattern
  that matches an `hlfir.elemental` and its `hlfir.assign`, store-to-load
  forwarding, and LICM of loads out of the region.
- Merging two redundant asserts means merging or inlining regions. Hoisting
  one means moving its whole region, or splitting it.
- Several checks on the same statement nest regions inside each other.

The guarded-result form gives the same guarantee at the granularity of a
value: guarding a loop's extent protects every access in the loop, without
moving the loop into a region.

## 4. Check categories and default policy

| Kind | Meaning | Default | Option |
|---|---|---|---|
| `status` | Allocation status required by the standard (ALLOCATE/DEALLOCATE, assignment to an unallocated or disassociated left-hand side) and allocation failure | Always on, all optimization levels | Engineering option to drop, for staging |
| `conformance` | Shape or size agreement that the runtime checks when not inlined | On (same behavior as `-O0`) | `-fno-check=conformance` (name to be decided) |
| `argument` | Intrinsic argument values the runtime checks (DIM range, RESHAPE SHAPE values) | On | same mechanism |
| `bounds` | Subscript, section, substring, and character-dummy-length checks | Off | `-fcheck=bounds` |
| `pointer` | Dereference of a disassociated POINTER, an unallocated ALLOCATABLE, or an absent OPTIONAL | Off | `-fcheck=pointer` |
| `do` | DO loop with zero step | Off | `-fcheck=do` |
| `bits` | Bit-intrinsic position and shift ranges | Off | `-fcheck=bits` |
| `divide` | Integer MOD/MODULO with `P==0` | Off | existing `-fcheck-integer-mod-zero-divisor` |

Gating:

- Default-on checks are always emitted. The assert lowering (or an earlier
  cleanup pass) drops them by category. Dropping replaces the results with the
  guarded operands.
- Opt-in checks are only emitted when enabled, so they cost nothing otherwise.
  The flags reach lowering and the passes through a module attribute, as
  `-fcheck-integer-mod-zero-divisor` does today (`fir.check_integer_mod_zero_divisor`).
  A single `fir.runtime_checks` attribute listing the enabled kinds would
  scale better.
- `-fcheck=` currently exists only as a gfortran pass-through in
  `clang/include/clang/Options/FlangOptions.td`. It would become a Flang
  option with gfortran-compatible sub-options.

## 5. Checks to emit

Each table gives the condition in IR terms, the runtime message to reuse, the
insertion point, and what to guard. `ext(x,d)` is the extent of dimension `d`
(0-based) from `hlfir::genExtentsVector` or `fir.box_dims`. `null(p)` is a null
test of a `fir.box_addr` result.

### 5.1 Checks lost by HLFIR and FIR inlining (`status`, `conformance`, `argument`)

These checks exist in the runtime today and disappear at `-O1` and above.

#### Assignment and copy

| ID | Operation | Condition (`%ok`) | Runtime message | Inserted in | Guard |
|---|---|---|---|---|---|
| A1 | Array assignment, no realloc | for each `d`: `ext(lhs,d) == ext(rhs,d)`. Skip for `temporary_lhs` and for assigns produced by `SeparateAllocatableAssign` (conform by construction; mark them with an attribute). | `Assign: mismatching element counts in array assignment (to %jd, from %jd)` (`RT/assign.cpp:453-461`; `AssignSimple` at `:921-928`) | `IHA` `InlineHLFIRAssignConversion`, before `hlfir::genNoAliasArrayAssignment` | Loop extents from `deduceOptimalExtents` |
| A2 | Assignment to a POINTER or non-realloc ALLOCATABLE left-hand side | `!null(lhs) \|\| size(lhs) == 0` | `Assign: left-hand side variable is neither allocated nor allocatable` (`RT/assign.cpp:371-374`) | `IHA` `InlineHLFIRAssignConversion`; `OB` `ElementalAssignBufferization`, `EvaluateIntoMemoryAssignBufferization` | Left-hand-side box |
| A3 | Elemental expression assigned in place | for each `d`: `ext(elemental,d) == ext(lhs,d)` | as A1 | `OB` `ElementalAssignBufferization` rewrite, where `rhsExtents` and `lhsExtents` are computed (`:463-473`); it currently assumes conformance (`:273-279`) | Loop extents |
| A4 | `hlfir.eval_in_mem` (inlined MATMUL and others) written directly into the left-hand side | `size(lhs) == product(ext(shape))`; per dimension for strict conformance | as A1 | `OB` `tryUsingAssignLhsDirectly` (`:632-713`) | Left-hand-side address passed to the region |
| A5 | Scalar broadcast to an unallocated allocatable (realloc) or disassociated pointer | `!null(lhs)` | `Assign: mismatched ranks (%jd != %jd) in assignment to unallocated allocatable` (`RT/assign.cpp:361-370`), or the existing inline text `array left hand side must be allocated when the right hand side is a scalar` (`MutableBox.cpp:947-954`) | `OB` `BroadcastAssignBufferization` (`:514-612`); it fires even with `realloc` | Left-hand-side box |
| A6 | Inline (re)allocation for assignment | `!null(newStorage)` after `fir.allocmem` | `Memory allocation failed` / `AssignSimple: allocation failed (stat=%jd)` (`RT/assign.cpp:130`, `:1050`) | `SAA` (runs at every optimization level), `IHA` `InlineAllocatableExprAssignConversion`, `C2F` scalar allocatable path (`:123-150`), all through `fir::factory::genReallocIfNeeded` / `allocateAndInitNewStorage` | New storage address |

Not lost: `InlineCopyInConversion` replaces `ShallowCopyDirect`, whose checks
are internal, because the temporary is built from the variable's own shape.

Note on A1: the runtime only compares **element counts**. A `2x3 = 3x2`
assignment passes the runtime and gives a wrong result in element order. The
inline loop nest would access out of bounds, so the assert must compare
**per-dimension** extents. That is stricter than the runtime, and it is
exactly F2018 10.2.1.2 conformance. The message then also names the
dimension (see the A3 example in section 5.4).

#### Reductions, MATMUL, DOT_PRODUCT

| ID | Operation | Condition | Runtime message | Inserted in | Guard |
|---|---|---|---|---|---|
| A7 | DOT_PRODUCT | `ext(a,0) == ext(b,0)` | `DOT_PRODUCT: SIZE(VECTOR_A) is %jd but SIZE(VECTOR_B) is %jd` (`RT/dot-product.h:63-67`) | `SHI` `DotProductConversion::genProductExtent`, before `deduceOptimalExtents`; `SI` DOT_PRODUCT at the call site (the generated body always loops over the first vector) | Loop extent |
| A8 | MATMUL | `ext(a, rank(a)-1) == ext(b,0)` | `MATMUL: unacceptable operand shapes (%jdx%jd, %jdx%jd)` and the vector forms (`RT/matmul.h:272-293`) | `SHI` `MatmulConversion::genResultShape`, before `deduceOptimalExtents` | Inner extent |
| A9 | MATMUL_TRANSPOSE | `ext(a,0) == ext(b,0)` | `MATMUL-TRANSPOSE: unacceptable operand shapes (%jdx%jd, %jdx%jd)` (`RT/matmul-transpose.h:201-209`) | same | Inner extent |
| A10 | MASK of SUM, PRODUCT, MAXVAL, MINVAL, MAXLOC, MINLOC | `!present(mask) \|\|` for each `d`: `ext(mask,d) == ext(array,d)`, with the extents read inside `fir.if present` | `Incompatible array arguments to %s: dimension %jd of ARRAY has extent %jd but MASK has extent %jd` (`RT/tools.cpp:97-104`) | `SHI` `ReductionAsElementalConverter::convert` (`:1142-1266`); `SI` `simplifyMinMaxlocReduction` | ARRAY extents (loop bounds) |
| A11 | Non-constant DIM on a rank-1 argument of ALL, ANY, COUNT, MAXLOC, MINLOC | `dim == 1` | `%s: bad DIM=%jd for ARRAY with rank 1` (`RT/reduction.cpp:256`, `RT/tools.cpp:345`) | `SHI` `AllAnyAsElementalConverter`, `CountAsElementalConverter`, `MinMaxlocAsElementalConverter`; `SI` `simplifyLogicalDim1Reduction`, `simplifyMinMaxlocReduction` | None (DIM is not used for addressing) |

Not lost:

- Partial reductions with a non-constant DIM are not inlined (`getConstDim`).
- A constant DIM is checked by semantics.
- MAXLOC and MINLOC with a BACK argument are not inlined by HLFIR.
- FIR-level SUM, MAXVAL and COUNT only fire without DIM (and, for SUM and
  MAXVAL, without MASK).
- For SUM, PRODUCT, MAXVAL and MINVAL on rank-1 arrays, the runtime path already
  ignores DIM, so an assert there would be a new check (kind `argument`).

#### Transformational intrinsics

| ID | Operation | Condition | Runtime message | Inserted in | Guard |
|---|---|---|---|---|---|
| A12 | CSHIFT and EOSHIFT with an array SHIFT | for each `k`, with `j` the matching ARRAY dimension (skipping DIM): `ext(shift,k) == ext(array,j)` | `%s: on dimension %jd, SHIFT= has extent %jd but ARRAY= has extent %jd` (`RT/transformational.cpp:48-53`) | `SHI` `ArrayShiftConversion`, after the ARRAY extents are computed, before `hlfir.elemental` or `hlfir.eval_in_mem` | Result shape extents |
| A13 | EOSHIFT with an array BOUNDARY | `!present(boundary) \|\|` same per-dimension test | `EOSHIFT: BOUNDARY= has extent %jd on dimension %jd but must conform with extent %jd of ARRAY=` (`RT/transformational.cpp:630-643`) | same | Result shape extents |
| A14 | EOSHIFT on CHARACTER with BOUNDARY | `len(boundary) == len(array)` | `EOSHIFT: BOUNDARY= has element byte length %jd, but ARRAY= has length %jd` (`RT/transformational.cpp:625-629`, `:700-704`) | same; the inline path pads or truncates instead of failing | None |
| A15 | RESHAPE: SHAPE values | for each `i`: `shape(i) >= 0` | `RESHAPE: bad value for SHAPE(%jd)=%jd` (`RT/transformational.cpp:818-821`) | `SHI` `ReshapeAsElementalConversion`, after the SHAPE loads, before `fir.shape` (`:3130-3137`) | SHAPE extents fed to `fir.shape` |
| A16 | RESHAPE: enough elements | `product(shape) <= size(array) \|\| (present(pad) && size(pad) > 0)` | `RESHAPE: not enough elements, need %jd but only have %jd` (`RT/transformational.cpp:831-836`) | same | same |

A16 also prevents an unsigned division by zero in the inline index
computation when ARRAY or PAD is empty (`SHI:3262-3266`).

Not lost: TRANSPOSE (rank only, static), character comparison
(`hlfir.cmpchar`), and INDEX. Their runtime entry points have no user-facing
checks.

### 5.2 Checks missing in code generated inline during lowering

| ID | Kind | Operation | Condition | Message | Inserted in | Guard |
|---|---|---|---|---|---|---|
| B1 | `status` | ALLOCATE of an allocated allocatable ([#223287](https://github.com/llvm/llvm-project/pull/223287)) | `null(box)` | `the object '%s' is already allocated` (runtime stat text: `Base address is not null`) | `flang/lib/Lower/Allocatable.cpp`, `AllocateStmtHelper`, before `genInlinedAllocation` (`:562-574`) | None needed (the program terminates before the leak is observable) |
| B2 | `status` | DEALLOCATE of an unallocated allocatable | `!null(box)` | `the object '%s' is not allocated` (runtime: `Base address is null`) | same file, inline branch of `genDeallocate`, before `genFreemem` (`:1015-1020`) | None needed (`free(NULL)` is harmless) |
| B3 | `divide` | Integer MOD/MODULO, `P==0` | `p != 0` | `MOD with P==0` / `MODULO with P==0` | `IC` `genIntegerZeroDivisorCheck`: replace `fir.if`; also add it to the UNSIGNED path (`genMod`, `genModulo`), which has none | `p` (the `arith.remsi` operand) |
| B4 | migrate | NEAREST `S==0`, IEEE `RADIX/=2`, STORAGE_SIZE of an unallocated polymorphic, realloc with scalar RHS | existing conditions | existing messages | `IC` `genNearest`, `checkRadix`, `genStorageSize`; `MutableBox.cpp` `genReallocIfNeeded` | Operand of the protected op |
| B5 | `bits` | BTEST, IBCLR/IBSET, IBITS, ISHFTC, MASKL/MASKR, MVBITS (out of range gives poison); ISHFT and SHIFTL/SHIFTR/SHIFTA/DSHIFTL/DSHIFTR (clamped, so silently accepted) | `0 <= pos < bit_size`, `pos + len <= bit_size`, and similar | gfortran `-fcheck=bits` style | `IC` per intrinsic | Shift or position operand |

B1 and B2 are required by the standard and should be unconditional. The
runtime path already checks them (`RT/allocatable.cpp:142-143`, `:198-199`);
`-use-alloc-runtime` works around the problem today.

Also note: the CSHIFT shift normalization calls `genModulo` with the extent
hidden behind a `select`, so with `-fcheck-integer-mod-zero-divisor` every
CSHIFT gets a dead `MODULO with P==0` check (`SHI` `normalizeShiftValue`).
Constant folding of the assert, or emitting `arith.remsi` directly there,
fixes it.

### 5.3 New opt-in checks (`bounds`, `pointer`, `do`)

`inb(i, lb, ext)` is the single unsigned test `(i - lb) <u ext`. It is
equivalent to `lb <= i <= lb + ext - 1` and false when `ext == 0`.

| ID | Kind | Check | Condition | Inserted in | Guard | Notes |
|---|---|---|---|---|---|---|
| C1 | `bounds` | Array element subscript | for each `d`: `inb(i_d, lb_d, ext_d)`; for the last dimension of an assumed-size array, only `i >= lb` | `X2H` `HlfirDesignatorBuilder::visit(ArrayRef)`, scalar-subscript branch, after `genSubscript` (`:737-743`) | Subscript value | Skip constant subscripts (checked by semantics). Keep the legacy `dimension(1)` dummy idiom working. |
| C2 | `bounds` | Section triplet `lo:hi:st` | `st != 0 && (n == 0 \|\| (inb(lo) && inb(last)))`, `last = lo + (n-1)*st` | same function, triplet branch, before `Triplet{lb, ub, stride}` (`:713-736`) | `lo`, `hi`, `st` | Check `last`, not `hi` (`a(1:11:3)` is valid for `ub=10`). Zero-size sections are exempt. |
| C3 | `bounds` | Vector subscript elements | `inb(v(k), lb, ext)` | `X2H` `createVectorSubscriptElementAddrOp`, after the element load (`:995-1003`) | Loaded subscript | O(n). A MINVAL/MAXVAL pre-check is cheaper but loses the index in the message. |
| C4 | `bounds` | Substring `s(lo:hi)` | `hi < lo \|\| (lo >= 1 && hi <= len)` | `X2H` `gen(Substring)`, before the length is replaced (`:549-555`) | `lo`, `hi` | Must be done before `ConvertToFIR`, which drops `hi`. |
| C5 | `bounds` | Character dummy length | `passedLen >= declaredLen` | `flang/lib/Lower/ConvertVariable.cpp`, dummy mapping (non-box `:3129-3142`, box `:2898-2939`) | Dummy base address | Skip bare-reference dummies (no length) and absent OPTIONAL. |
| C6 | `pointer` | POINTER dereference | `!null(box_addr(load p))` | Context-aware call sites, **not** inside `hlfir::derefPointersAndAllocatables`: `X2H` `visit(SymbolRef)` (`:648-653`), the pointer-component paths (`:808-809`, `:889`), and `flang/lib/Lower/ConvertCall.cpp` for a non-optional, non-pointer dummy (`:1497-1498`) | Loaded box or base address | Several callers of `derefPointersAndAllocatables` dereference a possibly null box on purpose (a disassociated actual passed to an OPTIONAL dummy counts as absent, F2018 15.5.2.12). |
| C7 | `pointer` | ALLOCATABLE dereference | same as C6 | same sites | same | Exempt realloc assignment, `ALLOCATED`, `MOVE_ALLOC`, and actuals passed to ALLOCATABLE or OPTIONAL dummies. A zero-size allocation has a non-null address, so "allocated" means "non-null". |
| C8 | `pointer` | Use of an absent OPTIONAL dummy | `fir.is_present %x` | Part references (`X2H` `visit(SymbolRef)`), value uses, actuals passed to non-optional dummies (`ConvertCall.cpp`, `genIsPresentIfArgMaybeAbsent`) | Declare result at the use site | Guard per use, not once at entry (uses under `if (present(x))` are legal). Exempt `PRESENT` and forwarding to OPTIONAL dummies. |
| C9 | `do` | DO loop with zero step | `step != 0` | `flang/lib/Lower/Bridge.cpp`, after the step is computed (`:3094-3099`) | Step value | A zero step currently divides by zero in the trip-count computation. |

Out of scope for now: the explicit-shape dummy size versus the actual (callee
only has an address), and checks of undefined POINTER status.

### 5.4 IR examples

**A7, DOT_PRODUCT** (`SHI` `DotProductConversion`):

```
%da:3 = fir.box_dims %a, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
%db:3 = fir.box_dims %b, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
%eq   = arith.cmpi eq, %da#1, %db#1 : index
%n    = fir.assert %eq, conformance,
          "DOT_PRODUCT: SIZE(VECTOR_A) is %jd but SIZE(VECTOR_B) is %jd"
          values(%da#1, %db#1 : index, index) guard(%da#1 : index)
%r = fir.do_loop %i = %c1 to %n step %c1 iter_args(%acc = %zero) -> (f32) {
  %ea = hlfir.designate %a (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %eb = hlfir.designate %b (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  ...
}
```

**A3, elemental assignment** (`OB` `ElementalAssignBufferization`):

```
%e = hlfir.elemental %shr unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> { ... }
hlfir.assign %e to %x#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
// becomes
%dx:3 = fir.box_dims %x#0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
%eq   = arith.cmpi eq, %n, %dx#1 : index
%n_ok = fir.assert %eq, conformance,
          "Assign: mismatching extents on dimension %jd in array assignment (to %jd, from %jd)"
          values(%c1, %dx#1, %n : index, index, index) guard(%n : index)
fir.do_loop %i = %c1 to %n_ok step %c1 unordered { ...inlined elemental body; store into x(i)... }
```

When the elemental shape comes from `x` itself (`x = x + 1.0`), the two
extents are the same SSA value and the assert folds away.

**A10, SUM with an OPTIONAL MASK** (`SHI` `ReductionAsElementalConverter`):

```
%present = fir.is_present %mask : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
%m:2 = fir.if %present -> (i1, index) {
  %dm:3 = fir.box_dims %mask, %c0 : (...) -> (index, index, index)
  %eq   = arith.cmpi eq, %dm#1, %na : index
  fir.result %eq, %dm#1 : i1, index
} else {
  fir.result %true, %na : i1, index
}
%na_ok = fir.assert %m#0, conformance,
           "Incompatible array arguments to SUM: dimension %jd of ARRAY has extent %jd but MASK has extent %jd"
           values(%c1, %na, %m#1 : index, index, index) guard(%na : index)
// reduction loop over 1..%na_ok, MASK read at the same index
```

**A15/A16, RESHAPE without PAD** (`SHI` `ReshapeAsElementalConversion`):

```
%s1  = fir.load %p1 : !fir.ref<i32>
%nn  = arith.cmpi sge, %s1, %c0_i32 : i32
%s1i = fir.convert %s1 : (i32) -> index
%da:3 = fir.box_dims %a, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
%enough = arith.cmpi ule, %s1i, %da#1 : index
%ok  = arith.andi %nn, %enough : i1
%ext = fir.assert %ok, argument, "RESHAPE: bad SHAPE= or not enough elements in SOURCE="
         guard(%s1i : index)
%shape = fir.shape %ext : (index) -> !fir.shape<1>
%r = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> { ... }
```

Two separate asserts give the two exact runtime messages. One combined assert
gives a smaller IR.

**B1, ALLOCATE status** (`Allocatable.cpp`):

```
%bx   = fir.load %a#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
%addr = fir.box_addr %bx : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
%allocated = fir.is_present %addr : (!fir.heap<!fir.array<?xi32>>) -> i1   // non-null test
%free = arith.xori %allocated, %true : i1
fir.assert %free, status, "the object 'array' is already allocated"
%mem  = fir.allocmem !fir.array<?xi32>, %n {uniq_name = "_QFEarray.alloc"}
```

**C1, array element `a(i)`** (`X2H` `HlfirDesignatorBuilder`):

```
%d:3 = fir.box_dims %a#0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
%off = arith.subi %i, %d#0 : index
%ok  = arith.cmpi ult, %off, %d#1 : index
%i_ok = fir.assert %ok, bounds, "subscript %jd is out of bounds for dimension 1 of 'a' (%jd:%jd)"
          values(%i, %d#0, %ub : index, index, index) guard(%i : index)
%e = hlfir.designate %a#0 (%i_ok) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
```

**C6, POINTER dereference `p(i)`**:

```
%bx  = fir.load %p#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
%adr = fir.box_addr %bx : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
%ok  = fir.is_present %adr : (!fir.ptr<!fir.array<?xf32>>) -> i1
%bx_ok = fir.assert %ok, pointer, "POINTER 'p' is not associated" guard(%bx : !fir.box<!fir.ptr<!fir.array<?xf32>>>)
%e = hlfir.designate %bx_ok (%i) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
```

## 6. Where asserts are created, optimized and lowered

Pipeline (`flang/lib/Optimizer/Passes/Pipelines.cpp`), with the `fir.assert`
steps added:

| Stage | Pass (existing, unless marked new) | Optimization level | `fir.assert` role |
|---|---|---|---|
| Lowering | `flang/lib/Lower` (designators, ALLOCATE, DEALLOCATE, DO, intrinsics in `IC`) | all | Create B1-B5 and C1-C9 |
| HLFIR | `SimplifyHLFIRIntrinsics` (twice) | O1+ | Create A7-A16 |
| HLFIR | `InlineElementals`, `SeparateAllocatableAssign` | all | Create A6 (`SeparateAllocatableAssign`) |
| HLFIR | CSE, canonicalize | O1+ | Fold constant asserts; merges the conditions |
| HLFIR | `OptimizedBufferization`, `InlineHLFIRAssign` (twice), `InlineHLFIRCopy` (O3) | O1+ (O0 variants for device) | Create A1-A6 |
| HLFIR | `BufferizeHLFIR`, `ConvertHLFIRtoFIR` | all | Nothing (the op is FIR) |
| FIR | CSE, canonicalize, then **new: redundant-assert elimination** | O1+ | Dominance-based merging (section 3.4) |
| FIR | `SimplifyIntrinsics` | O1+ | Create A7, A10, A11 at the call site |
| FIR | flang LICM, extended | O1+ | Hoist invariant asserts (section 7) |
| FIR | **new: redundant-assert elimination** (second run) | O1+ | Merge asserts that hoisting made comparable |
| FIR | `SimplifyFIROperations`, `CFGConversion` | all | Nothing |
| Codegen | `FIRToLLVMLowering` | all | Drop disabled kinds; lower the rest to a branch plus noreturn call (section 3.5) |

At `-O0`, only lowering, `InlineElementals`, `SeparateAllocatableAssign`, and
the device variants of `InlineHLFIRAssign` create asserts, and none of the
optimizing steps run. The lowering pattern must work on unoptimized IR.

## 7. Optimization interactions

- **No interference with memory optimizations.** The assert writes only the
  non-addressable runtime resource (section 3.2).
- **Hoisting.** FIR LICM hoists only pure ops and loads, so asserts stay in
  the loop, and so do the operations depending on their guarded results. A
  dedicated rule is needed: move a loop-invariant assert to before the loop
  when the loop runs at least once (LICM already proves this for loads) and
  the assert is executed on every iteration. Its users can then be hoisted
  too. For a possibly-zero trip count, hoist `cond || tripCount == 0`.
- **Range checks for subscripts (C1).** For an affine subscript
  `s*i + c` in `fir.do_loop %i = lo to hi step st`, replace the per-iteration
  check with checks on the first and last executed values. Two variants:
  - *Predication*: report the failure before the loop. Legal (the program is
    non-conforming), but output written before the failing iteration is lost.
  - *Versioning*: choose between an unchecked fast loop and the original
    checked loop, reusing the cloning infrastructure of `fir::LoopVersioning`.
    Exact messages, and the fast path vectorizes. Recommended for
    `-O2 -fcheck=bounds`.
- **Vectorization.** After lowering, an assert inside a loop is an early exit
  to a noreturn call, which the LLVM loop vectorizer does not handle. The
  checks in sections 5.1 and 5.2 (A and B) are loop-invariant and placed
  before the loop nests, so they cost nothing per element. Only the checks in
  section 5.3 (C) can end up in hot loops.
- **Never check compiler-generated accesses.** Element accesses in loops
  created by `BufferizeHLFIR` and the inlining passes are in bounds by
  construction once the A checks hold. That is why bounds checks belong in
  `HlfirDesignatorBuilder` and not in a pass over all `hlfir.designate`, or in
  `fir.array_coor` codegen.
- **Device code.** Per-element checks in kernels cost registers and block
  unrolling and vectorized loads. For `-fcheck=bounds` in offloaded loops,
  prefer range checks before the kernel launch, and keep in-kernel checks for
  non-affine subscripts.

## 8. Suggested implementation order

1. `fir.assert` with `FortranRuntimeResource`, verifier, canonicalization,
   lowering, the runtime entry with values, and a `noreturn` declaration.
2. B1 and B2 (standard-required, one compare per statement). This supersedes
   the current version of
   [#223287](https://github.com/llvm/llvm-project/pull/223287).
3. A7-A9 and A1-A5 (out-of-bounds risks in the most common inlined code).
4. A10-A16 and A6.
5. Migrate the existing checks (B3, B4) and add the UNSIGNED MOD check.
6. Redundant-assert elimination and LICM support.
7. `-fcheck=` plumbing, then C9, C1, C2, C4, C5, then C6-C8, then C3 and B5.
8. Range checks and versioning for C1.

## 9. Open questions

1. Message formatting: a fixed number of `int64` values, or a variadic runtime
   entry? Do we keep the runtime's exact messages, or switch to one message
   style for inline checks?
2. Should disabled default-on checks become `llvm.intr.assume` (more
   optimization, but a false condition is then undefined behavior) or be
   dropped?
3. Option names and defaults for `conformance` and `argument`. Turning them
   on by default matches the runtime, but changes `-O2` behavior on
   non-conforming programs (they now stop instead of producing wrong results
   or crashing).
4. Allocation failure (A6): check after each `fir.allocmem`, or give
   `fir.allocmem` a "fail on null" attribute handled in codegen?
5. Device runtime availability of the error entry point in each offload
   configuration (CUDA, OpenACC, OpenMP offload).

## Appendix: runtime issues found during the survey

- `RT/character.cpp`, `CharacterCompare`: `yChars` uses `shift<char>` instead
  of `shift<CHAR>`, so KIND=2 and KIND=4 array comparisons read past the end
  of the second operand.
- `RT/tools.cpp`, `CheckIntegerKind`: the format arguments are passed in the
  wrong order.
- `RT/reduction.cpp`: the rank-1 DIM check of ALL, ANY, COUNT and PARITY
  accepts `DIM=0`.
- `ReportFatalUserError` is not in the OpenMP offload API group, unlike
  `Abort`.
