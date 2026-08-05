# Temporary flatten + depth verification cases

Throwaway sandbox for eyeballing the flatten desugaring (AST / IR / runtime).
Not part of any PR — delete freely. The committed regression tests live under
`clang/test/OpenMP/flatten*` and `openmp/runtime/test/transform/flatten/*`.

`CLANG=build/bin/clang`, `CLANGXX=build/bin/clang++`, run from the repo root.

| File | Case | What to look for |
|------|------|------------------|
| `01_default_depth2.cpp` | default (no clause) = depth 2 | i64 `.flatten.iv`, `mul nsw`, `sdiv`+`srem` |
| `02_depth3_full.cpp` | `depth(3)` full flatten | 3 `.flatten.iv.N`, two-level mixed radix |
| `03_depth2_partial.cpp` | `depth(2)` on 3-deep nest | 2 flatten IVs, **inner `%k` loop stays** |
| `04_depth1_identity.cpp` | `depth(1)` identity | i32 IV, **no `srem`**, direct store |
| `05_rangefor_template.cpp` | range-for + template `depth(K)` | ast-print round-trips; instantiation |
| `end_result_check.cpp` | runtime semantic check | prints `OK`, exit 0 (order+set preserved) |

## Commands

AST (round-trip original loops) and the desugared node:
```
$CLANG -cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=61 -ast-print FILE
$CLANG -cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=61 -ast-dump  FILE
```

IR (the div/mod arithmetic that implements the transform):
```
$CLANG -cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=61 -emit-llvm -o - FILE
```

Range-for / template need the driver (libstdc++ + template instantiation):
```
$CLANGXX -fopenmp -fopenmp-version=61 -Xclang -ast-print -fsyntax-only 05_rangefor_template.cpp
```

Runtime end-result (must print OK, exit 0):
```
$CLANGXX -fopenmp -fopenmp-version=61 -O0 end_result_check.cpp -o /tmp/erc && /tmp/erc
```
