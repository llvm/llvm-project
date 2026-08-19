# Code Coverage in LLVM-libc

(code_coverage)=

This document describes how to configure, generate, and view code coverage and Modified Condition / Decision Coverage (MC/DC) reports for LLVM-libc.

---

## Overview

Code coverage measures the proportion of source code executed during testing. In LLVM-libc, coverage metrics identify untested edge cases, prevent regressions across supported architectures, and provide verification evidence for safety-critical systems.

### Coverage Modes

* **Statement and Branch Coverage:**  
  Measures line execution and verifies whether conditional branches evaluated to both true and false paths.

* **Modified Condition / Decision Coverage (MC/DC):**  
  Evaluates boolean conditions within compound decisions (such as `if (A && B)`). Verifies that each individual sub-condition is tested with both true and false values and independently affects the outcome of the enclosing decision. Required by safety-critical standards such as DO-178C (aviation) and ISO 26262 (automotive).

---

## Prerequisites & Toolchain Setup

Generating coverage reports requires Clang, LLVM profile tools, CMake, and Ninja:

* **Compiler:** Clang 18 or later (Clang 21 or later is required for MC/DC instrumentation).
* **LLVM Utilities:** Matching major versions of `llvm-profdata` and `llvm-cov`.
* **Build System:** CMake 3.28+ and Ninja.

### Toolchain Installation (Debian/Ubuntu)

Install the required compiler, tooling, and build utilities:

```bash
sudo apt install clang-21 llvm-21-tools lld-21 ninja-build cmake
```

### Configuring Tool Links

Because Debian and Ubuntu install LLVM tools with version suffixes (such as `llvm-profdata-21`), you can configure unversioned aliases using either of the following methods:

#### Method A: User-Level Setup (Recommended - No `sudo` required)

```bash
mkdir -p ~/.local/bin
ln -sf $(which llvm-profdata-21 2>/dev/null || which llvm-profdata) ~/.local/bin/llvm-profdata
ln -sf $(which llvm-cov-21 2>/dev/null || which llvm-cov) ~/.local/bin/llvm-cov
export PATH="$HOME/.local/bin:$PATH"
```

#### Method B: System-Wide Alternatives (Requires `sudo`)

```bash
sudo update-alternatives --install /usr/bin/llvm-profdata llvm-profdata /usr/bin/llvm-profdata-21 100
sudo update-alternatives --install /usr/bin/llvm-cov llvm-cov /usr/bin/llvm-cov-21 100
```

:::{note}
Compiling the full suite of unit tests with coverage instrumentation encompasses ~10,000 target nodes. On multi-core workstations, parallel compilation completes in a few minutes. On resource-constrained systems, virtual machines with 2–4 cores, or cold builds without compiler caching, the initial build may take up to 30 minutes.
:::

---

## Cleaning Profile Counters

Before running a new coverage test pass, remove existing `.profraw` and `.profdata` files to avoid merging stale profiling data:

```bash
find build-cov -name "libc_cov_*.profraw" -delete 2>/dev/null || true
rm -f build-cov/libc_full.profdata libc_full.profdata profraw_list.txt
```

---

## How to Run Standard Coverage (Statement and Branch)

Standard coverage records line execution and branch direction metrics across all LLVM-libc entrypoints and support routines.

### 1. CMake Configuration

Configure LLVM-libc with `-DLLVM_LIBC_ENABLE_COVERAGE=ON`:

```bash
cmake -G Ninja -S runtimes -B build-cov \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_RUNTIMES=libc \
  -DLLVM_LIBC_FULL_BUILD=ON \
  -DLLVM_LIBC_ENABLE_COVERAGE=ON \
  -DLIBC_ENABLE_MCDC=OFF \
  -DLIBC_TEST_UNIT_TEST_ONLY=ON \
  -DLIBC_TEST_SKIP_DEATH_TESTS=ON
```

### 2. Build Unit Tests

Compile the test suite:

```bash
ninja -k 0 -C build-cov libc-unit-tests || true
```

### 3. Run Unit Tests

Execute the compiled unit test binaries in parallel across available CPU cores:

```bash
# Clean stale counters
find build-cov -name "libc_cov_*.profraw" -delete 2>/dev/null || true
rm -f libc_full.profdata profraw_list.txt

# Run all test binaries with isolated PID profile names
export LLVM_PROFILE_FILE="libc_cov_%p.profraw"
(cd build-cov && find libc/test -type f -executable -name "*__build__" | xargs -P $(nproc) -I {} sh -c '{} > /dev/null 2>&1 || true')
```

### 4. Merge Profile Counters

Resolve the matching profile merge tool and aggregate the emitted raw counters:

```bash
# Auto-detect matching llvm-profdata based on active Clang version
CLANG_MAJOR=$(clang --version | sed -n 's/.*version \([0-9]*\).*/\1/p')
LLVM_PROFDATA=$(which llvm-profdata-$CLANG_MAJOR 2>/dev/null || which llvm-profdata)

find . build-cov -name "libc_cov_*.profraw" > profraw_list.txt
$LLVM_PROFDATA merge -sparse --input-files=profraw_list.txt -o libc_full.profdata
```

### 5. View Coverage Reports

Collect test binary object references and discover the coverage viewer:

```bash
CLANG_MAJOR=$(clang --version | sed -n 's/.*version \([0-9]*\).*/\1/p')
LLVM_COV=$(which llvm-cov-$CLANG_MAJOR 2>/dev/null || which llvm-cov)

EXECUTABLES=($(find build-cov -type f -executable -name "*__build__"))
OBJECTS=("${EXECUTABLES[@]:1}")
OBJECTS=("${OBJECTS[@]/#/-object=}")
```

#### Terminal Summary Table

Display a directory-by-directory coverage report in the terminal:

```bash
$LLVM_COV report \
  -instr-profile=libc_full.profdata \
  "${EXECUTABLES[0]}" "${OBJECTS[@]}" \
  --show-branch-summary \
  -ignore-filename-regex=".*(test|utils).*"
```

#### Interactive HTML Dashboard

Generate a browsable HTML site with source file coverage drill-downs:

```bash
$LLVM_COV show \
  -format=html \
  -output-dir=coverage_html \
  -instr-profile=libc_full.profdata \
  "${EXECUTABLES[0]}" "${OBJECTS[@]}" \
  --show-directory-coverage \
  --show-branches=count \
  --compilation-dir=. \
  --path-equivalence="$PWD,." \
  -ignore-filename-regex=".*(test|utils).*"

# Open in browser
xdg-open coverage_html/index.html
```

---

## How to Run Modified Condition / Decision Coverage (MC/DC)

MC/DC instrumentation captures condition-level truth tables for compound logical expressions in addition to statement and branch metrics.

### 1. CMake Configuration

Enable MC/DC instrumentation by adding `-DLIBC_ENABLE_MCDC=ON`:

```bash
cmake -G Ninja -S runtimes -B build-cov \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_RUNTIMES=libc \
  -DLLVM_LIBC_FULL_BUILD=ON \
  -DLLVM_LIBC_ENABLE_COVERAGE=ON \
  -DLIBC_ENABLE_MCDC=ON \
  -DLIBC_TEST_UNIT_TEST_ONLY=ON \
  -DLIBC_TEST_SKIP_DEATH_TESTS=ON
```

### 2. Build Unit Tests

Compile the test binaries:

```bash
ninja -k 0 -C build-cov libc-unit-tests || true
```

### 3. Run Unit Tests

Execute the compiled unit test binaries in parallel:

```bash
# Clean stale counters
find build-cov -name "libc_cov_*.profraw" -delete 2>/dev/null || true
rm -f libc_full.profdata profraw_list.txt

# Run all test binaries with isolated PID profile names
export LLVM_PROFILE_FILE="libc_cov_%p.profraw"
(cd build-cov && find libc/test -type f -executable -name "*__build__" | xargs -P $(nproc) -I {} sh -c '{} > /dev/null 2>&1 || true')
```

### 4. Merge Profile Counters

Merge the raw counters into an indexed profile dataset:

```bash
# Auto-detect matching llvm-profdata based on active Clang version
CLANG_MAJOR=$(clang --version | sed -n 's/.*version \([0-9]*\).*/\1/p')
LLVM_PROFDATA=$(which llvm-profdata-$CLANG_MAJOR 2>/dev/null || which llvm-profdata)

find . build-cov -name "libc_cov_*.profraw" > profraw_list.txt
$LLVM_PROFDATA merge -sparse --input-files=profraw_list.txt -o libc_full.profdata
```

### 5. View MC/DC Coverage Reports

Collect test binary object references:

```bash
CLANG_MAJOR=$(clang --version | sed -n 's/.*version \([0-9]*\).*/\1/p')
LLVM_COV=$(which llvm-cov-$CLANG_MAJOR 2>/dev/null || which llvm-cov)

EXECUTABLES=($(find build-cov -type f -executable -name "*__build__"))
OBJECTS=("${EXECUTABLES[@]:1}")
OBJECTS=("${OBJECTS[@]/#/-object=}")
```

#### Terminal Summary Table with MC/DC Metrics

Display statement, branch, and MC/DC decision coverage percentages in the terminal:

```bash
$LLVM_COV report \
  -instr-profile=libc_full.profdata \
  "${EXECUTABLES[0]}" "${OBJECTS[@]}" \
  --show-branch-summary \
  --show-mcdc-summary \
  -ignore-filename-regex=".*(test|utils).*"
```

#### Interactive HTML Dashboard with MC/DC Analysis

Generate an interactive HTML dashboard containing MC/DC decision breakdown tables and line-by-line coverage:

```bash
$LLVM_COV show \
  -format=html \
  -output-dir=coverage_mcdc_html \
  -instr-profile=libc_full.profdata \
  "${EXECUTABLES[0]}" "${OBJECTS[@]}" \
  --show-directory-coverage \
  --show-branches=count \
  --show-mcdc \
  --show-mcdc-summary \
  --compilation-dir=. \
  --path-equivalence="$PWD,." \
  -ignore-filename-regex=".*(test|utils).*"

# Open in browser
xdg-open coverage_mcdc_html/index.html
```

---

## Running Coverage for a Single Target

To quickly inspect coverage for an individual entrypoint (e.g. `strlen`) without building the entire library:

```bash
# 1. Build the specific test binary
ninja -C build-cov libc.test.src.string.strlen_test.__unit__.__build__

# 2. Run the test binary
export LLVM_PROFILE_FILE="libc_cov_%p.profraw"
./build-cov/libc/test/src/string/libc.test.src.string.strlen_test.__unit__.__build__

# 3. Merge profiles
CLANG_MAJOR=$(clang --version | sed -n 's/.*version \([0-9]*\).*/\1/p')
LLVM_PROFDATA=$(which llvm-profdata-$CLANG_MAJOR 2>/dev/null || which llvm-profdata)
LLVM_COV=$(which llvm-cov-$CLANG_MAJOR 2>/dev/null || which llvm-cov)

find . build-cov -name "libc_cov_*.profraw" > profraw_list.txt
$LLVM_PROFDATA merge -sparse --input-files=profraw_list.txt -o libc_single.profdata

# 4. View MC/DC truth table in terminal
TEST_BIN="./build-cov/libc/test/src/string/libc.test.src.string.strlen_test.__unit__.__build__"
$LLVM_COV show \
  -instr-profile=libc_single.profdata \
  "$TEST_BIN" \
  --show-branches=count \
  --show-mcdc \
  libc/src/string/strlen.cpp
```

---

## Interpreting Results

Understanding coverage metrics helps developers assess test completeness, identify uncovered edge cases, and author targeted unit tests.

### Terminal Summary Metrics

When executing `llvm-cov report`, the terminal output summarizes coverage across files and directories:

* **Region Coverage:**  
  Measures execution of discrete Abstract Syntax Tree (AST) expression sub-blocks (such as the body of an `if` statement or ternary expressions). A lower region coverage than line coverage indicates partially executed expressions on lines that were counted as hit.

* **Line Coverage:**  
  Tracks physical source lines executed during the test run. Unexecuted lines represent functions, conditional branches, or error recovery handlers that were never invoked.

* **Branch Coverage:**  
  Evaluates conditional branch outcomes. If a branch indicates `50%` coverage, the condition was only ever evaluated in one direction (for example, always `True`), leaving the alternative path (`False`) untested.

* **MC/DC Coverage:**  
  Reports the percentage of compound boolean decisions where every atomic sub-condition was demonstrated to independently determine the final decision outcome.

### HTML Dashboard and Source Inspection

The interactive HTML dashboard (`coverage_html/index.html`) provides line-by-line visual inspection of source implementations:

#### Line Execution Highlights

* **Green Lines:** Source code executed by tests. The margin integer indicates execution count.
* **Red Lines:** Unexecuted code that requires additional unit test coverage.

#### Branch Markers

Conditional statements display branch hit counts inline in the format `[True: N, False: M]`. An entry of `[True: 10, False: 0]` indicates that the conditional expression was never evaluated as False during testing. Adding a unit test where the condition evaluates to False resolves this gap.

#### MC/DC Truth Tables and Condition Diagnostics

When `--show-mcdc` is enabled, `llvm-cov` renders a boolean truth table directly below compound decisions:

```
   19|    517|  if (c < 0 || c > 255)
  ------------------
  |  Executed MC/DC Test Vectors:
  |     C1, C2    Result
  |  1 { F,  F  = F      }
  |  2 { T,  -  = T      }
  |
  |  C1-Pair: covered (1, 2)
  |  C2-Pair: not covered
  |  MC/DC Coverage for Decision: 50.00%
  ------------------
```

##### Understanding the Truth Table

1. **Identify the Conditions:**  
   In `if (c < 0 || c > 255)`, condition **C1** is `c < 0` and **C2** is `c > 255`.

2. **Inspect Executed Vectors:**  
   * **Vector 1 (`F, F = F`):** Tested with an in-range value (e.g. `c = 100`). Both C1 and C2 evaluated to False, producing a False outcome.
   * **Vector 2 (`T, - = T`):** Tested with a negative value (e.g. `c = -1`). C1 evaluated to True, which immediately satisfied the `if` statement (C2 was short-circuited `-`).

3. **Evaluate Coverage Status:**  
   * **`C1-Pair: covered (1, 2)`:** Verified. Comparing Vector 1 and Vector 2 proves that toggling C1 alone flips the overall decision outcome.
   * **`C2-Pair: not covered`:** Missing. C2 was never tested in a state where it independently caused the `if` condition to become True.

4. **How to Fix the Gap:**  
   Add a unit test case with `c = 256`. This evaluates C1 as False and C2 as True (`3 { F, True = True }`), forming the missing independence pair `(1, 3)` for C2 and achieving 100% MC/DC coverage.
