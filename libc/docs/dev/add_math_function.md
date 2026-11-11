(add_math_function)=

# Implementing and Adding a Math Function

This guide describes the file layout, implementation conventions, caveats, and
testing requirements for adding or refactoring math functions in LLVM-libc.

## 1. Math Function Architecture & Layout

Every math function in LLVM-libc separates its **header-only core
implementation** from its **public C entrypoint** and **shared internal wrapper**
so the same implementation can be linked into `libm.a`, consumed header-only by
other LLVM subprojects (`libc++`, offloading runtimes), and tested per
evaluation variant:

```text
libc/
├── include/math.yaml                         # Public C header specification
├── src/
│   ├── __support/
│   │   ├── FPUtil/                           # Generic FP, DoubleDouble, bit utilities
│   │   └── math/
│   │       ├── <func>.h                      # Dispatch header (selects *_eval)
│   │       ├── <func>_double_eval.h          # Double-precision evaluation
│   │       ├── <func>_float_eval.h           # Single-precision evaluation (optional)
│   │       ├── <func>_integer_eval.h         # Integer/soft-float evaluation (optional)
│   │       ├── <func>_utils.h                # Shared family/function tables & helpers
│   │       ├── <func>_accurate_128.h         # 128-bit fallback pass (optional)
│   │       └── <func>_accurate_256.h         # 256-bit fallback pass (optional)
│   └── math/
│       ├── <func>.h                          # Internal entrypoint declaration
│       ├── generic/<func>.cpp                # LLVM_LIBC_FUNCTION wrapper
│       └── <arch>/<func>.cpp                 # Arch-specific override (optional)
├── shared/
│   ├── math.h                                # Umbrella header for shared math
│   └── math/<func>.h                         # Header-only export in shared::
└── test/
    ├── shared/shared_math_test.cpp           # Header-only coexistence smoke test
    └── src/math/
        ├── smoke/<func>_test.cpp             # MPFR-free smoke tests
        ├── <func>_test.cpp                   # MPFR unit tests
        ├── exhaustive/<func>_test.cpp        # Parallel exhaustive / worst-case tests
        └── performance_testing/<func>_perf.cpp
```

### Modular `src/__support/math/` Organization

Simple functions with a single implementation (such as basic bit manipulations
or compact `float16`/`bfloat16` functions) can place their implementation
directly inside `libc/src/__support/math/<func>.h` under
`LIBC_NAMESPACE_DECL::math`.

Functions with multiple evaluation strategies, shared family utilities, or
multi-tier precision passes are split into modular headers:

1. **Dispatch Header (`<func>.h`)**:
   Selects an evaluation namespace via `LIBC_MATH_<FUNC>_IMPL` based on target
   hardware features (`LIBC_TARGET_CPU_HAS_FPU_FLOAT`,
   `LIBC_TARGET_CPU_HAS_FPU_DOUBLE`, `LIBC_TARGET_CPU_HAS_FMA_*`,
   `LIBC_TARGET_ARCH_IS_GPU`) and `LIBC_MATH_HAS_*` optimization flags. Guard
   the default selection with `#ifndef LIBC_MATH_<FUNC>_IMPL` and use
   `LIBC_MATH_IMPL_IS` (from `src/__support/macros/optimization.h`) so external
   `-DLIBC_MATH_<FUNC>_IMPL=...` overrides only `#include` the selected header:

   ```cpp
   #ifndef LIBC_MATH_EXPF_IMPL
   #if defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT) &&                                \
       defined(LIBC_MATH_HAS_SKIP_ACCURATE_PASS) &&                             \
       (defined(LIBC_MATH_HAS_SMALL_TABLES) ||                                  \
        defined(LIBC_MATH_HAS_INTERMEDIATE_COMP_IN_FLOAT))
   #define LIBC_MATH_EXPF_IMPL float_eval
   #elif !defined(LIBC_TARGET_CPU_HAS_FPU_DOUBLE) &&                            \
       defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) &&                      \
       defined(LIBC_MATH_HAS_NO_EXCEPT) && defined(LIBC_MATH_HAS_NO_ERRNO)
   #define LIBC_MATH_EXPF_IMPL integer_eval
   #else
   #define LIBC_MATH_EXPF_IMPL double_eval
   #endif
   #endif // !LIBC_MATH_EXPF_IMPL

   #if LIBC_MATH_IMPL_IS(LIBC_MATH_EXPF_IMPL, float_eval)
   #include "src/__support/math/expf_float_eval.h"
   #elif LIBC_MATH_IMPL_IS(LIBC_MATH_EXPF_IMPL, integer_eval)
   #include "src/__support/math/expf_integer_eval.h"
   #elif LIBC_MATH_IMPL_IS(LIBC_MATH_EXPF_IMPL, double_eval)
   #include "src/__support/math/expf_double_eval.h"
   #else
   #error "Unsupported LIBC_MATH_EXPF_IMPL"
   #endif // LIBC_MATH_IMPL_IS

   namespace LIBC_NAMESPACE_DECL {
   namespace math {

   using LIBC_MATH_EXPF_IMPL::expf;

   } // namespace math
   } // namespace LIBC_NAMESPACE_DECL

   #undef LIBC_MATH_EXPF_IMPL
   ```

2. **Evaluation Variant Headers (`*_{double,float,integer}_eval.h`)**:
   - `<func>_double_eval.h` (`math::double_eval`): Double-precision evaluation
     (either standalone for `float` functions or Tier-1 `DoubleDouble` fast path
     for `double` functions).
   - `<func>_float_eval.h` (`math::float_eval` or `math::float_eval_cr_rn`):
     Single-precision evaluation for targets with hardware float FPU when
     reduced-size or fast math modes are requested.
   - `<func>_integer_eval.h` (`math::integer_eval` /
     `math::static_rounding::integer_eval`): Pure integer fixed-point evaluation
     for soft-float targets and compile-time static rounding modes.

3. **Shared Utilities & Multi-Precision Passes**:
   - `<func>_utils.h` / `<family>_utils.h` (and mode-specific
     `<family>_double_utils.h`, `<family>_float_utils.h`,
     `<family>_integer_utils.h`): Shared lookup tables, polynomial evaluators,
     subnormal/denormal rounding helpers, and exceptional-case handlers.
   - `<func>_accurate_128.h` and `<func>_accurate_256.h`: Cold 128-bit and
     256-bit fixed-point (`Frac64`, `Frac128`, `Frac256`, `DyadicFloat`) passes
     invoked only when the fast path's Ziv rounding test cannot guarantee
     correct rounding.

### Math Optimization Flags (`LIBC_CONF_MATH_OPTIMIZATIONS`)

Build configs can set `LIBC_CONF_MATH_OPTIMIZATIONS` (propagated as the bitmask
macro `LIBC_MATH` in `src/__support/macros/optimization.h`):

| Flag | Derived Macro | Effect |
| :--- | :--- | :--- |
| `LIBC_MATH_SKIP_ACCURATE_PASS` | `LIBC_MATH_HAS_SKIP_ACCURATE_PASS` | Skips multi-precision fallback passes and exact-exception tables; targets {math}`< 1` ULP accuracy with compact code size and predictable latency. |
| `LIBC_MATH_SMALL_TABLES` | `LIBC_MATH_HAS_SMALL_TABLES` | Uses smaller lookup tables (typically paired with `LIBC_MATH_SKIP_ACCURATE_PASS`) and outlines shared helpers via `LIBC_MATH_SMALL_TABLES_INLINE` to minimize `.text` + `.rodata` footprint. |
| `LIBC_MATH_NO_ERRNO` | `LIBC_MATH_HAS_NO_ERRNO` | Omits `errno` updates (`EDOM`, `ERANGE`), avoiding TLS/PLT calls on cold paths. |
| `LIBC_MATH_NO_EXCEPT` | `LIBC_MATH_HAS_NO_EXCEPT` | Omits IEEE-754 floating-point exception raising (`FE_INVALID`, `FE_DIVBYZERO`, `FE_OVERFLOW`, `FE_UNDERFLOW`). |
| `LIBC_MATH_INTERMEDIATE_COMP_IN_FLOAT` | `LIBC_MATH_HAS_INTERMEDIATE_COMP_IN_FLOAT` | Prefers single-precision intermediate arithmetic for `float` functions when hardware float FPU is available. |
| `LIBC_MATH_ASSUME_ROUND_NEAREST_ONLY` | `LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY` | Assumes round-to-nearest, ties-to-even (`FE_TONEAREST`) without dynamic rounding-mode queries. |
| `LIBC_MATH_FAST` | *(all of the above except `INTERMEDIATE_COMP_IN_FLOAT`)* | Default embedded/baremetal fast preset combining small tables, no accurate pass, no `errno`/`fenv`, and round-to-nearest. |

---

## 2. Implementation Caveats & Best Practices

### Namespace Hygiene & Avoiding Symbol Collisions

`libc/shared/math.h` includes **every** `src/__support/math/*.h` header into a
**single translation unit** (verified by `shared_math_test.cpp`), and functions
such as `pow` transitively include `exp2`, `exp10`, and `exp_utils`. Any
duplicate constant, struct, or helper name placed directly in a shared
namespace will cause redefinition errors or silent overload-resolution bugs.

Follow a strict 3-layer namespace convention:

1. **Keep `math::double_eval`, `math::float_eval`, and `math::integer_eval`
   clean**: Prefer placing only the top-level dispatch targets (`exp`, `pow`,
   `sinf`, ...) directly inside `*_eval`. Since C math function names are
   globally unique, entrypoint names never collide.
2. **Scope private & family-shared symbols in `<func>_internal` or
   `<family>_internal`**:
   - Put function-private tables, polynomials, and helpers in `<func>_internal`
     (e.g., `pow_internal`, `tanf_internal`).
   - Put family-shared utilities in `<family>_internal` (e.g., `exp_internal`,
     `sincosf_utils_internal`).
   - **Never** write `using namespace <foo>_internal;` at header or namespace
     scope. Place `using namespace` inside function bodies only, and explicitly
     qualify cross-family helper calls so dependencies remain obvious.
3. **Promote generic primitives to `fputil::` and keep constants in
   family `<family>_utils.h` headers**:
   - Generic floating-point bit masks, DoubleDouble arithmetic, and fixed-point
     splitting helpers belong in `src/__support/FPUtil/`.
   - Mathematical constants and lookup tables belong in their owning family's
     `*_utils.h` header (e.g., `exp_utils.h`, `log_utils.h`, `sincos_utils.h`)
     rather than a catch-all constants file, using explicit precision/bit-width
     suffixes (e.g., `LOG_2_HI_12` vs `LOG_2_HI_43`) when multiple splits exist.

### Header Hygiene & Bazel `layering_check` / `parse_headers`

- Every support header must define a unique header guard, `#include`
  `"src/__support/common.h"` and `"src/__support/macros/config.h"`, and
  directly `#include` every header whose declarations it uses.
- Avoid `#include "hdr/stdint_proxy.h"` in `src/__support/math/*.h` when
  `"src/__support/common.h"` or `"src/__support/FPUtil/FPBits.h"` is already
  included.
- Give each `.h` file in `src/__support/math/` its own granular
  `add_header_library` target in `libc/src/__support/math/CMakeLists.txt` and a
  matching `libc_support_library` target in
  `utils/bazel/llvm-project-overlay/libc/BUILD.bazel` listing **exact direct
  dependencies** (both unconditional and conditional `#include`s). Always verify
  with Bazel `--features=layering_check --features=parse_headers`.

### Floating-Point Evaluation, Rounding Modes, `errno`, and Cold Paths

- **Split chained floating-point expressions**: Write one floating-point
  operation per statement with descriptive intermediate variable names. Long
  chained expressions (`a * b + c * d + e`) invite compiler contraction or
  reassociation that can invalidate error bounds in `DoubleDouble`, FastTwoSum,
  or range-reduction arithmetic.
- **`fputil::multiply_add` vs `fputil::fma`**: Use `fputil::multiply_add(a, b, c)`
  when FMA is an optimization or guarded by `#ifdef LIBC_TARGET_CPU_HAS_FMA_*`;
  it compiles to hardware FMA when available and `a * b + c` otherwise. Only
  call `fputil::fma` when software-emulated FMA is intentionally required on
  non-FMA targets (which is much slower than a dedicated non-FMA code path).
- **Keep non-FMA bit masking in FP/SIMD registers**: On targets without FMA,
  prefer `fputil::fp_and`, `fputil::fp_or`, `fputil::fp_xor`, and
  `fputil::scale_exp2` (`src/__support/FPUtil/fp_bitwise.h`) over Veltkamp
  splitting or `cpp::bit_cast<uint64_t>` round-trips between general-purpose and
  FP registers.
- **All 4 rounding modes, `errno`, and exceptions**: Unless
  `LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY` is active, implementations must be
  correctly rounded in `FE_TONEAREST`, `FE_DOWNWARD`, `FE_UPWARD`, and
  `FE_TOWARDZERO`, and handle FTZ/DAZ subnormal modes cleanly. Use
  `fputil::set_overflow<T>`, `fputil::set_underflow<T>`,
  `fputil::set_errno_if_required`, and `fputil::raise_except_if_required`
  (`except_value_utils.h` / `FEnvImpl.h`) so `NO_ERRNO` and `NO_EXCEPT` builds
  compile out side effects automatically.
- **Outline cold paths without stack spills**: Mark exceptional, subnormal, and
  multi-precision fallback functions `LIBC_COLD LIBC_NOINLINE LIBC_INLINE`.
  Return results (or a small 2-field struct) **by value** in registers instead
  of passing pointers or non-const references to hot-path local variables, which
  forces stack frame setup and register spills on the fast path.
- **Compiler compatibility**: Ensure implementations compile warning-free under
  both Clang and GCC (`>= 12.2`), and keep all lines within 80 columns
  (`clang-format --dry-run -Werror`).

---

## 3. Step-by-Step Checklist to Add a Math Function

### Step 1: Registration & Public Specification

1. Add the function specification to `libc/include/math.yaml`.
2. Enable `libc.src.math.<func>` under `TARGET_LIBM_ENTRYPOINTS` (inside the
   appropriate `LIBC_TYPES_HAS_FLOAT16` / `LIBC_TYPES_HAS_FLOAT128` guard when
   applicable) in `entrypoints.txt` for each supported platform and
   architecture:
   - **Linux** (`x86_64`, `aarch64`, `arm`, `riscv`; `i386` and `power` also
     have `entrypoints.txt` files):
     - `libc/config/linux/x86_64/entrypoints.txt`
     - `libc/config/linux/aarch64/entrypoints.txt`
     - `libc/config/linux/arm/entrypoints.txt`
     - `libc/config/linux/riscv/entrypoints.txt`
     - `libc/config/linux/i386/entrypoints.txt`
     - `libc/config/linux/power/entrypoints.txt`
   - **Baremetal** (`aarch64`, `arm`, `hexagon`, `riscv`):
     - `libc/config/baremetal/aarch64/entrypoints.txt`
     - `libc/config/baremetal/arm/entrypoints.txt`
     - `libc/config/baremetal/hexagon/entrypoints.txt`
     - `libc/config/baremetal/riscv/entrypoints.txt`
   - **Darwin / macOS** (`aarch64`, `x86_64`):
     - `libc/config/darwin/aarch64/entrypoints.txt`
     - `libc/config/darwin/x86_64/entrypoints.txt`
   - **FreeBSD** (`x86_64`):
     - `libc/config/freebsd/x86_64/entrypoints.txt`
   - **GPU** (`amdgpu`, `nvptx`, `spirv`):
     - `libc/config/gpu/amdgpu/entrypoints.txt`
     - `libc/config/gpu/nvptx/entrypoints.txt`
     - `libc/config/gpu/spirv/entrypoints.txt`
   - **Windows**:
     - `libc/config/windows/entrypoints.txt`
   - **UEFI**:
     - `libc/config/uefi/entrypoints.txt`
3. Mark the function as implemented in the status tables in
   `libc/docs/headers/math/index.md`.

### Step 2: Core Implementation & Entrypoint Wrapper

1. Add the header-only implementation (`<func>.h` plus any `*_eval.h`,
   `*_utils.h`, and `*_accurate_*.h` headers) in `libc/src/__support/math/` and
   register each header with `add_header_library` in
   `libc/src/__support/math/CMakeLists.txt`.
2. Declare `LIBC_NAMESPACE_DECL::<func>` in `libc/src/math/<func>.h` and add
   `add_math_entrypoint_object(<func>)` in `libc/src/math/CMakeLists.txt`.
3. Define the `LLVM_LIBC_FUNCTION` wrapper in
   `libc/src/math/generic/<func>.cpp` forwarding to `math::<func>`, and add the
   `add_entrypoint_object` target in `libc/src/math/generic/CMakeLists.txt`
   (plus any architecture-specific override in `libc/src/math/<arch>/<func>.cpp`).

### Step 3: Shared Math Header & Bazel Overlay

1. Create `libc/shared/math/<func>.h` exporting `using math::<func>;` inside
   `LIBC_NAMESPACE_DECL::shared`, and `#include "math/<func>.h"` in
   `libc/shared/math.h`.
2. Add a call in `libc/test/shared/shared_math_test.cpp` and add
   `libc.src.__support.math.<func>` to `libc/test/shared/CMakeLists.txt`.
3. Add matching `libc_support_library` targets and `libc_math_function` rule in
   `utils/bazel/llvm-project-overlay/libc/BUILD.bazel`.

### Step 4: MPFR Wrapper Support (if needed)

If the mathematical operation is not yet supported by `MPFRWrapper`:

1. Add an enum entry to `LIBC_NAMESPACE::testing::mpfr::Operation` in
   `libc/utils/MPFRWrapper/MPFRUtils.h`.
2. Implement the corresponding `MPFRNumber` method and dispatch case in
   `libc/utils/MPFRWrapper/MPFRUtils.cpp`.

---

## 4. Testing Strategy by Precision Tier

Every math function requires **smoke tests** (`libc/test/src/math/smoke/`) and
**MPFR unit tests** (`libc/test/src/math/`), registered in both `CMakeLists.txt`
and `utils/bazel/llvm-project-overlay/libc/test/src/math/{,smoke/}BUILD.bazel`:

- **Smoke tests (`smoke/<func>_test.cpp`)** do not depend on MPFR and run on all
  targets (including baremetal, UEFI, and GPUs). Test {math}`\pm 0`, subnormals,
  {math}`\pm 1`, {math}`\pm\infty`, quiet and signaling NaNs, overflow/underflow
  `errno` and floating-point exceptions, and FTZ/DAZ subnormal modes.
- **Variant parameterization**: When a function has evaluation variants
  (`double_eval`, `float_eval`, `integer_eval`), parameterize both smoke and
  unit tests so `LIBC_NAMESPACE::<func>` (`Default`) and each
  `LIBC_NAMESPACE::math::<mode>_eval::<func>` variant are tested directly on
  every CI run regardless of which variant `<func>.h` selects by default.

Beyond smoke tests, verification scales according to the floating-point width:

### 16-Bit Floating-Point (`float16`, `bfloat16`)

For unary 16-bit functions (`*f16`, `*bf16`), the entire type has only
{math}`2^{16} = 65{,}536` bit patterns. **Test all {math}`2^{16}` inputs
exhaustively across all 4 rounding modes directly inside the regular unit test**
(`libc/test/src/math/<func>_test.cpp`), which runs in milliseconds:

```cpp
// Range: [0, +Inf] and [-Inf, -0] for float16 (0x7c00 / 0xfc00)
// or bfloat16 (0x7f80 / 0xff80).
TEST_F(LlvmLibcSinBf16Test, PositiveRange) {
  for (uint16_t v = 0x0000U; v <= 0x7f80U; ++v) {
    bfloat16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sin, x,
                                   LIBC_NAMESPACE::sinbf16(x), 0.5);
  }
}

TEST_F(LlvmLibcSinBf16Test, NegativeRange) {
  for (uint16_t v = 0x8000U; v <= 0xff80U; ++v) {
    bfloat16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sin, x,
                                   LIBC_NAMESPACE::sinbf16(x), 0.5);
  }
}
```

(Adjust tolerance to `TOLERANCE + 0.5` when `LIBC_MATH_HAS_SKIP_ACCURATE_PASS`
relaxes rounding to {math}`< 1.5` ULP.)

### Single-Precision (`float`, 32-Bit)

For 32-bit `float` functions (`*f`), combine three test suites:

1. **Smoke tests** (`libc/test/src/math/smoke/<func>f_test.cpp`) for special
   values, exceptions, `errno`, and FTZ/DAZ behavior across all variants.
2. **Unit tests** (`libc/test/src/math/<func>f_test.cpp`) checking known tricky
   inputs, overflow/underflow thresholds, and representative sample sweeps
   against MPFR across all 4 rounding modes and all `*_eval` variants.
3. **Full {math}`2^{32}` Exhaustive Tests**
   (`libc/test/src/math/exhaustive/<func>f_test.cpp` and `CMakeLists.txt`):
   Use `LlvmLibcUnaryOpExhaustiveMathTest` (and
   `LlvmLibcUnaryOpAgainstBaselineExhaustiveMathTest` /
   `LlvmLibcStaticallyRoundedUnaryOpExhaustiveMathTest` for `float_eval` and
   `static_rounding` variants) from `exhaustive_test.h` to sweep **all
   {math}`2^{32}` single-precision bit patterns** across `[0, +Inf]` and
   `[-Inf, -0]` in all 4 rounding modes using multithreaded shards. Run
   exhaustive tests locally whenever adding a new `float` function or modifying
   polynomial/range-reduction constants to verify correct rounding (or ULP
   bounds) and identify any exceptional inputs.

### Double-Precision (`double`, 64-Bit)

Because sweeping all {math}`2^{64}` bit patterns is infeasible, double-precision
functions combine targeted unit tests with **CORE-MATH** worst-case
verification:

1. **Smoke & MPFR Unit Tests** (`smoke/<func>_test.cpp` and `<func>_test.cpp`):
   Test special numbers, subnormal inputs and outputs, FTZ/DAZ modes, exact
   algebraic/midpoint rounding boundaries, known hard-to-round inputs, and
   strided/random samples across all 4 rounding modes for both `Default` and
   `double_eval`.
2. **CORE-MATH Worst-Case & Special-Value Verification**:
   Test against the [CORE-MATH](https://core-math.gitlabpages.inria.fr/)
   repository's precomputed worst-case (`--worst`, typically {math}`10^6+`
   hardest-to-round points) and boundary/midpoint (`--special`) suites across
   all 4 rounding modes (`--rndn`, `--rndz`, `--rndu`, `--rndd`) on both FMA and
   non-FMA targets:

   ```bash
   # 1. Compile the entrypoint with LIBC_COPT_PUBLIC_PACKAGING so it exports the
   #    unprefixed C symbol (repeat with -march=x86-64-v2 to test non-FMA):
   clang++ -O3 -DNDEBUG -std=gnu++17 -march=x86-64-v3 \
     -DLIBC_NAMESPACE=__llvm_libc_git -DLIBC_COPT_PUBLIC_PACKAGING \
     -D__LIBC_USE_BUILTIN_ISNAN -DLIBC_COMPILER_HAS_STDC_FENV_ACCESS \
     -Ilibc -Ibuild/libc -isystem build/libc/include \
     -c libc/src/math/generic/<func>.cpp -o /tmp/<func>.o
   ar rcs /tmp/lib<func>.a /tmp/<func>.o

   # 2. Run CORE-MATH worst-case and special checks across all 4 rounding modes
   #    (pass -DCORE_MATH_UNDERFLOW_BEFORE on architectures like x86-64 that
   #    detect tininess before rounding):
   git clone https://gitlab.inria.fr/core-math/core-math.git /tmp/core-math
   cd /tmp/core-math
   EXTRA_CFLAGS="-DCORE_MATH_UNDERFLOW_BEFORE" LIBM=/tmp/lib<func>.a \
     ./check.sh --worst <func>
   EXTRA_CFLAGS="-DCORE_MATH_UNDERFLOW_BEFORE" LIBM=/tmp/lib<func>.a \
     ./check.sh --special <func>
   ```

   ```{note}
   `libc/test/src/math/exhaustive/worst_case_test.h` already provides the
   in-tree harness (see `exhaustive/sin_test.cpp`, which loads `.wc` files from
   `LIBC_WORST_CASE_DIR` or `libc/test/src/math/exhaustive/`), and CORE-MATH's
   worst-case datasets for remaining double-precision functions will be wired
   into `libc/test/src/math/exhaustive/` in the near future.
   ```

### Performance Tests & CORE-MATH Benchmarking

#### 1. In-Tree Performance Tests

Add differential benchmarks in
`libc/test/src/math/performance_testing/<func>_perf.cpp` (and register the
target in `libc/test/src/math/performance_testing/CMakeLists.txt`) to compare
against the system `libm` across specific bit-pattern intervals using
`PerfTest::run_perf_in_range` (e.g., normal positive/negative inputs, near-zero
or near-one inputs, subnormal outputs, and overflow/underflow domains).

#### 2. CORE-MATH Performance Benchmarking (`perf.sh`)

CORE-MATH's `perf.sh` benchmarks **reciprocal throughput** (default) and
**dependent-chain latency** (`PERF_ARGS="--latency"`) across three targets in
order: CORE-MATH, the system `libm`, and an external static archive passed via
`LIBM` (such as `build/libc/lib/libllvmlibc.a` from an overlay build, or the
standalone `/tmp/lib<func>.a` archive built above):

```bash
cd /tmp/core-math

# Reciprocal throughput (default CORE_MATH_PERF_MODE=perf uses `perf stat`;
# set CORE_MATH_PERF_MODE=rdtsc when perf counters are restricted):
CC=clang CFLAGS="-O3 -march=x86-64-v3" \
  CORE_MATH_PERF_MODE=rdtsc \
  CORE_MATH_LAUNCHER="taskset -c 2" \
  LIBM=/tmp/lib<func>.a \
  ./perf.sh <func>

# Latency (pass --latency via PERF_ARGS):
CC=clang CFLAGS="-O3 -march=x86-64-v3" \
  CORE_MATH_PERF_MODE=rdtsc \
  CORE_MATH_LAUNCHER="taskset -c 2" \
  PERF_ARGS="--latency" \
  LIBM=/tmp/lib<func>.a \
  ./perf.sh <func>
```

Useful `perf.sh` environment variables:
- `CORE_MATH_PERF_MODE`: `perf` (default; uses `perf stat -e cycles`, which may
  require `kernel.perf_event_paranoid <= 0`) or `rdtsc` (uses hardware cycle
  counters directly without `perf` permissions).
- `PERF_ARGS`: Extra flags forwarded to `./perf`, such as `--latency` (measures
  latency instead of reciprocal throughput) and `--rndn` / `--rndz` / `--rndu` /
  `--rndd` (selects rounding mode).
- `PERF_COUNT` and `PERF_REPEAT`: Sample array size (default `100000`) and
  repeat count (default `500`).
- `CORE_MATH_SIMPLE_STAT=1` / `CORE_MATH_QUIET=1`: Runs a single trial instead
  of 20 trials / prints only the minimum cycles per call.
- `CORE_MATH_LAUNCHER="taskset -c <cpu>"`: Pins execution to a single CPU core
  to reduce scheduling and frequency-scaling noise.
- `CORE_MATH_STD_NAME=<libc_symbol>`: Maps CORE-MATH's function name when the C
  symbol name in `LIBM` differs (for example, `bfloat16` functions use
  `sin_bf16` in CORE-MATH vs `sinbf16` in LLVM-libc:
  `CORE_MATH_STD_NAME=sinbf16 LIBM=... ./perf.sh sin_bf16`).

#### 3. Specifying or Updating Input Ranges in CORE-MATH

CORE-MATH separates its benchmark sampling ranges from its correctness test
suites as follows:

- **Performance benchmark ranges (`perf.sh`)**:
  Defined per mathematical function family (shared across `binary16`,
  `binaryb16`, `binary32`, `binary64`, `binary80`, and `binary128`) in:
  ```text
  src/generic/<short_name>/random_under_test.h
  ```
  where `<short_name>` is the base function name without a type suffix (e.g.,
  `src/generic/exp/random_under_test.h` for `expf16`, `expbf16`, `expf`, and
  `exp`; `src/generic/pow/random_under_test.h` for `powf` and `pow`):
  - **Univariate functions** implement `random_under_test(void)`. Many trig and
    elementary headers guard their bounds with `#ifndef XMIN` / `#ifndef XMAX`,
    allowing one-off range overrides without editing files:
    ```bash
    EXTRA_CFLAGS="-DXMIN=-0x1.0p-10 -DXMAX=0x1.0p-10" \
      PERF_ARGS="--latency" LIBM=/tmp/lib<func>.a ./perf.sh <func>
    ```
    To benchmark custom distributions (e.g., subnormal outputs, near-1 inputs,
    large arguments, or slow-path fallbacks), update `random_under_test(void)`
    in `src/generic/<short_name>/random_under_test.h` directly.
  - **Bivariate functions** (`pow`, `atan2`, `hypot`, etc.) implement two
    generators in `src/generic/<short_name>/random_under_test.h`:
    `random_under_test_0(void)` for the first argument and
    `random_under_test_1(void)` for the second argument.
- **Special / random correctness ranges (`./check.sh --special <func>`)**:
  Defined per format and function in:
  ```text
  src/binary{16,b16,32,64,80,128}/<short_name>/check_special.c
  ```
  Edit `check_special.c` to add targeted interval sweeps, subnormal/overflow
  threshold neighborhoods, or exact/midpoint grid tests (and override random
  trial counts via `EXTRA_CFLAGS="-DCORE_MATH_TESTS=..."`).
- **Worst-case tables (`./check.sh --worst <func>`)**:
  Stored as line-oriented hex-float inputs in:
  ```text
  src/binary{16,b16,32,64,80,128}/<short_name>/<func>.wc
  ```

---

## 5. Building and Running Tests

### CMake + Ninja (Runtimes Build)

`add_libc_testsuite` creates paired `<suite>` (build + run) and `<suite>-build`
(build-only) targets, while `add_fp_unittest` / `add_libc_test` creates a Ninja
runner target `<fq_target_name>` alongside its executable target
`<fq_target_name>.__build__` (or `<fq_target_name>.__build_only__` for
`BUILD_ONLY` suites such as `libc-math-exhaustive-tests`, so `check-libc` and
`libc-math-tests` compile exhaustive test binaries without executing them via
`lit`). When an entrypoint declares `FMA_OPT` / `ROUND_OPT` flags, non-FMA and
non-round-instruction test variants (`*.__NO_FMA_OPT`, `*.__NO_ROUND_OPT`,
`*.__NO_FMA_OPT.__NO_ROUND_OPT`) are generated automatically as well.

```bash
mkdir -p build && cd build
cmake ../runtimes -G Ninja \
  -DLLVM_ENABLE_RUNTIMES="libc" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++

# Build and run all libc tests via lit (compiles exhaustive tests via
# check-libc-build / libc-tests-build without running *.__build_only__):
ninja check-libc

# Run all math smoke tests (no MPFR required; add `-build` to compile only):
ninja libc-math-smoke-tests
ninja libc-math-smoke-tests-build

# Run all math smoke + MPFR unit tests (and build exhaustive test binaries):
ninja libc-math-tests
ninja libc-math-tests-build

# Build and run a specific smoke test or MPFR unit test (plus non-FMA variant):
ninja libc.test.src.math.smoke.<func>_test
ninja libc.test.src.math.<func>_test
ninja libc.test.src.math.<func>_test.__NO_FMA_OPT

# Run shared header-only coexistence tests:
ninja libc-shared-tests
ninja libc.test.shared.shared_math_test

# Build and run a specific exhaustive / worst-case test via its runner target:
ninja libc.test.src.math.exhaustive.<func>_test
ninja libc.test.src.math.exhaustive.<func>_test.__NO_FMA_OPT

# Or build the exhaustive executable only (`*.__build_only__`) to run manually
# (e.g. with LIBC_WORST_CASE_DIR or gtest/custom arguments):
ninja libc.test.src.math.exhaustive.<func>_test.__build_only__
./libc/test/src/math/exhaustive/libc.test.src.math.exhaustive.<func>_test.__build_only__

# Build all exhaustive tests without running them, or run the entire suite:
ninja libc-math-exhaustive-tests-build
ninja libc-math-exhaustive-tests

# Build and run differential performance benchmark (writes <func>_perf.log):
ninja libc.test.src.math.performance_testing.<func>_perf
./libc/test/src/math/performance_testing/libc.test.src.math.performance_testing.<func>_perf
```

### Bazel Overlay & Header Layering Verification

From `utils/bazel/`, verify strict header layering, standalone header parsing
across `LIBC_MATH` modes, and unit/smoke tests:

```bash
cd utils/bazel

# Verify header self-containment and dependency layering:
bazel build --repo_env=CC=clang \
  --features=layering_check --features=parse_headers \
  @llvm-project//libc:<func>

bazel build --repo_env=CC=clang \
  --features=layering_check --features=parse_headers \
  --copt=-DLIBC_MATH=LIBC_MATH_FAST \
  @llvm-project//libc:<func>

# Run smoke and MPFR unit tests:
bazel test --repo_env=CC=clang --@llvm-project//libc:mpfr=system \
  @llvm-project//libc/test/src/math/smoke:<func>_test \
  @llvm-project//libc/test/src/math:<func>_test
```
