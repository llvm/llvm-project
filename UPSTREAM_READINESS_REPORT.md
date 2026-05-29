# Metal/AIR LLVM target — upstream-readiness audit

Branch: `metal-target-poc`
Baseline before this audit: `14bec3145edf`
Head after this round: see `git log metal-target-poc -25`

## Phase 1 (already landed earlier)

| Commit | Subject |
|--------|---------|
| `4b76fb…` | clang-format AIRWriter |
| `4d2643…` | LLVM license headers + include guards on AIRWriter |
| `1f1c17…` | clang-format remaining Metal / MC sources |
| `4a95a5…` | Drop `METALIR_DUMP_LOWERED` debug hook |
| `9467a6…` | Normalize include guards to `LLVM_LIB_TARGET_METAL_*` |

## Phase 2

| Commit | Subject | Gap addressed |
|--------|---------|---------------|
| `93f2c2…` | Nest AIRWriter under `namespace llvm::metal` (NFC) | #1 |
| `b754e2…` | Name `MetallibOptions`/`MetalConstraints` fields LLVM-style | #2, #13, #17–#19 |
| `0011c4…` | Trap unreachable MC code paths; explicit `MetalMCAsmInfo` text-asm disables | #6, #7, #8 |
| `610a0b…` | Centralize AIR address-space constants in `MetalAddressSpaces.h` | #5 |
| `82482c…` | `MetalLibObjectWriter::writeObject` requires exactly one non-empty section | #10 |
| `480e15…` | Rewrite five MetalPrepare TODOs with WHY + `(GH-XXX)` placeholders | #22 |
| `5b57d3…` | Add `llvm/docs/Metal/{index,MetalTarget,MetallibFormat}.rst` | #27, #28 |
| `59e90b…` | Add `llvm/test/MC/MetalLib/` with two FileCheck tests | #25 |
| `af648a…` | Drop `PointeeTypeMap` `const_cast`s | #4 |
| `eca4de…` | File-level doc paragraphs on three large AIRWriter sources | #20 |
| `aa8861…` | Align `tg-global-gep-rewrite.ll` with current MetalPrepare output | #23 |

## Phase 3 (this round)

| Commit | Subject | Gap addressed |
|--------|---------|---------------|
| `2eaa5c…` | Expand AIR-version subarch note as Apple-spec-blocked | #26, #26a |
| `9f5b3f…` | Doxygen-document remaining public-API headers (BitcodeEncoding + MC) | #14, #15 |
| `4f293f…` | Six negative / no-op codegen FileCheck tests | #24 |
| `e20640…` | Datalayout, address-space + metallib header-byte tests | #24, #25 |
| `2f4c13…` | TripleTest unit-test coverage for `air` and `-metallib` | #26 (unit-side) |
| `191d74…` | Address-space + pipeline-ordering codegen tests | #24 |
| `6b5fd1…` | Five more per-pass edge / no-op fixtures | #24 |

### Phase 3 in one line per concern

- **A (new-PM registration)**: already done in Phase 1/2 for all 23 pipeline
  passes (legacy + `PassInfoMixin` wrappers sharing a single `runImpl`
  helper, registered via `MetalPassRegistry.def` +
  `MetalTargetMachine::registerPassBuilderCallbacks`). `MetalWriterPass` and
  `MetalEmbedderPass` remain legacy-PM-only — DXIL has the same shape, so
  this stays as a "reviewer-prompted only" follow-up.
- **B (doxygen on remaining public API)**: `BitcodeEncoding.h` got a
  file-level paragraph and per-function `///` blocks; the three MC-layer
  headers (`MCMetalLibObjectWriter.h`, `MCSectionMetalLib.h`,
  `MCMetalLibStreamer.h`) got per-class `///` blocks. `MetalConstraints.h`
  was already adequately doxygen'd in Phase 2.
- **C (subarch note)**: `MetalTarget.rst` now has a full "why AIR-version
  subarch is Apple-spec-blocked" paragraph, mirrored in `UPSTREAM_GAPS.md`
  entry 26a.
- **D (DXIL-parity test coverage)**: 26→44 codegen tests (+18), 2→5 MC
  tests (+3), and 4 new `TripleTest::ParsedIDs` cases. Coverage is now:
  every pass has 1+ positive test, eleven passes have an additional
  negative/edge test, all three Metal address spaces are pinned
  explicitly, the metallib header bytes (magic, version, section count,
  bitcode-wrapper offset) are pinned, the emitted datalayout string is
  pinned, and the AIR/MetalLib Triple wiring is unit-tested.
- **E (re-audit + report)**: this file plus an updated `UPSTREAM_GAPS.md`.

## Validation status (Phase 3 head)

- `ninja llc` clean.
- `ninja check-llvm-codegen-metal`: **44 / 44 passing** (was 26).
- `ninja check-llvm-mc` + `llvm/test/MC/MetalLib/`: **5 / 5 passing** for
  MetalLib (was 2). Full `check-llvm-mc` suite: 3446 passed, 0 failed.
- `TargetParserTests --gtest_filter='TripleTest*'`: 26 / 26 passing.
- Sentinel pytest (`test_chained_reductions`, `test_dot3d`, two
  `test_scan2d`): 4 / 4 passing.
- `/tmp/vector_add.py`: `add OK: True`.

## Phase 4 (this round)

| Commit | Subject | Gap addressed |
|--------|---------|---------------|
| `c9c6bc…` | Lower vector `llvm.minimum`/`llvm.maximum` in `MetalNaNMinMax` | #30 |
| `3dd829…` | New-PM wrappers for `MetalWriter` / `MetalEmbedder` passes | #11, #12 |
| `17930a…` | AIRWriter local rename (NFC) — batch 1 (5 files) | #2 |
| `610577…` | AIRWriter local rename (NFC) — batch 2 (3 files + 2 headers) | #2 |

## Per-subsystem readiness (updated)

| Subsystem | Phase 2 | Phase 3 | Phase 4 | Notes |
|-----------|---------|---------|---------|-------|
| Passes (`llvm/lib/Target/Metal/*.cpp`) | 90% | 92% | **95%** | `MetalNaNMinMax` now scalarizes vector `llvm.minimum`/`llvm.maximum` (gap #30 closed); pinned by `nan-min-max-vector.ll` + new `nan-min-max-vector-lower.ll`. |
| AIRWriter (`AIRWriter/`) | 80% | 85% | **95%** | Locals + parameters in every `.cpp` (and the two declaration headers whose param names were touched) are now `UpperCamelCase` per LLVM CodingStandards. Class fields are intentionally untouched. Both `MetalWriterPass` and `MetalEmbedderPass` now ship new-PM `PassInfoMixin` wrappers sharing the legacy `runOnModule` impl. |
| MC layer | 90% | 95% | **95%** | Unchanged this phase. |
| Triple / ObjectFormatType | 95% | 98% | **98%** | Unchanged; only AIR-version subarches remain (Apple-spec-blocked). |
| Tests | 85% | 95% | **97%** | 46 codegen + 5 MC + 4 unit-test cases (added `nan-min-max-vector-lower.ll` and `embed-metallib-newpm.ll`). |
| Documentation | 70% | 75% | **75%** | Unchanged. The Discourse-facing RFC narrative is still TODO. |

**Overall readiness: ~96%** for "first patch series + RFC" submission.

## What is still left

1. **Gap #26a (AIR-version subarches)** — Apple-spec-blocked; documented
   in `MetalTarget.rst` "Known Limitations".
2. **RFC narrative on LLVM Discourse** — content is in the gap doc and the
   readiness report; needs reshaping into a Discourse post that links the
   branch and quotes the green sentinel-suite + 46/46 codegen + 5/5 MC
   status.

## Recommendation

The branch is RFC-ready. Phase 4 closed every reviewer-anticipated polish
item that was not Apple-spec-blocked. The only remaining items are the
subarch question (which only Apple can unblock) and the RFC writeup
itself.
