# EmbeddedJIT Integration Tests

Integration tests for the EmbeddedJIT JIT compilation system.

## Quick Start

```bash
# Build and run all tests (from repo root)
./ejit_test/build.sh --run

# Build and run specific tests
./ejit_test/build.sh --run ejit_attr_test ejit_ptr_period_test

# Build only
./ejit_test/build.sh ejit_config_api_test
```

**Prerequisites**: a release build must exist:
```bash
# Native aarch64 (recommended):
./build.sh release aarch64     # → build_release_aarch64/ (clang + static libs)

# Or x86:
./build.sh debug x86           # → build_debug_x86/ (clang for compilation)
./build.sh release x86         # → build_release_x86/ (static libs for linking)
```

## Test Files

| Test | Description |
|------|-------------|
| `ejit_attr_test` | All 6 EJIT attribute types, multi-scenario (SPEC4 §2, §5) |
| `ejit_complex_test` | 4-dimension, multi-array, external cellIdx, early-return |
| `ejit_config_api_test` | Config, stats, cache, compile-mode API validation |
| `ejit_external_idx_test` | External cellIdx with multi-dim arrays |
| `ejit_jit_verify_test` | JIT correctness: constant folding, dead branch elimination |
| `ejit_lifecycle_test` | ejit_period_lc: deactivate/activate pairing |
| `ejit_multidim_test` | 2D multi-dim array with external cellIdx |
| `ejit_multiversion_test` | Multi-version JIT specialization per cellIdx |
| `ejit_nested_struct_test` | 2-level and 3-level nested struct may_const |
| `ejit_opt_level_test` | L1/L2/L3 optimization level validation |
| `ejit_perf_bench` | Performance benchmark: JIT compile time, cache hit latency |
| `ejit_ptr_period_test` | Pointer-type ejit_period_arr and ejit_period (NEW) |
| `ejit_trace_test` | Runtime trace: JIT dispatch, fallback, lifecycle hooks |

### External Dependency Tests (not in build.sh)

| Test | Dependency | Description |
|------|-----------|-------------|
| `ejit_zstd_bench` | libzstd | EJIT + zstd: per-compression-level JIT specialization |
| `ejit_zlib_bench` | system -lz | EJIT + zlib: per-compression-level JIT specialization |
| `ejit_stb_bench` | stb_image.h | EJIT + stb_image: per-channel-count JIT specialization (WIP) |

These are built manually; see the benchmark section below.

## IR Verification

After compilation, check these LLVM IR features:

| Check | grep command | Expected |
|-------|-------------|----------|
| Wrapper jit_entry block | `jit_entry:` | One per ejit_entry function |
| jit_fallback block | `jit_fallback:` | Original function body moved here |
| jit_dispatch block | `jit_dispatch:` | JIT-compiled function call |
| ejit_compile_or_get call | `ejit_compile_or_get` | In jit_entry block |
| may_const metadata | `ejit.may_const` | On marked field load instructions |
| Function/global metadata | `ejit.metadata` | `!"ejit_entry"` / `!"ejit_period"` / ... |
| Bitcode embedding | `ejit.bitcode` | `@__ejit_bitcode` in dedicated section |
| Auto-registration | `ejit_auto_register` | In `@llvm.global_ctors` |
| Auto symbol reg | `ejit_register_symbol` | External symbols auto-registered |
| Lifecycle deactivate | `ejit_deactivate_array` | At ejit_period_lc function entry |
| Lifecycle activate | `ejit_activate_array` | At ejit_period_lc function exits |

## Architecture

```
./build.sh release aarch64     # native AArch64 (build_release_aarch64/)
./build.sh release x86         # native x86 (build_release_x86/)
./build.sh debug x86           # debug clang (build_debug_x86/)
```

- **Compiler**: `build_<arch>/bin/clang` — with EJIT attributes enabled
- **Runtime**: `build_<arch>/lib/libLLVMEJIT.a` — release static library
- **Linker**: `build_<arch>/bin/ld.lld`
- **Lipo**: `ejit_test/lipo/ejit.o` — single relocatable .o replacing all LLVM .a files (~37 MB)

## Related Documents

- [SPEC4.md](../jit_design_doc/SPEC4.md) — requirements specification
- [PLAN4.md](../jit_design_doc/PLAN4.md) — architecture and pass design
- [CLANG_ATTR_DESIGN.md](../jit_design_doc/CLANG_ATTR_DESIGN.md) — Clang attribute implementation
