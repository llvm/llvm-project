# ClangIR Emit Points

`-cc1 -emit-cir` prints CIR after the CIR-to-CIR pipeline.
`-emit-cir=<keyword>` stops at an earlier point. A keyword names a condition
of the emitted CIR, not a pass.

| Keyword | Condition |
|---|---|
| `pristine` | CIRGen's output; no ClangIR pass has run. |

## ClangIR input

`-cc1` accepts CIR as input (`.cir` or `-x cir`) and runs the full CIR-to-CIR
pipeline on it, whichever point it was emitted at. The passes are safe to
rerun. `-clangir-disable-passes` skips the pipeline.
