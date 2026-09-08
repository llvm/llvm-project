(code_coverage)=

# Code Coverage

Code coverage is a software testing metric that measures the proportion of
source code executed while running an automated test suite. It provides insight
into test thoroughness by identifying untested functions, dead code paths, and
unexercised conditional branches across library entrypoints and internal
utilities.

LLVM-libc supports Modified Condition / Decision Coverage (MC/DC). MC/DC
evaluates compound boolean decisions composed of multiple sub-conditions (such
as `if (A && (B || C))`). Under MC/DC criteria, each individual boolean
condition must:
* Evaluate to both true and false across the test suite.
* Demonstrate that it can independently affect the outcome of the overall
  decision while other conditions remain fixed.

This provides rigorous structural verification for safety-critical algorithms
without requiring exhaustive testing of all 2<sup>n</sup> condition
permutations.

## Continuous Profiling Architecture

LLVM-libc uses Clang's [continuous profiling mode][clang-continuous]
(`-fprofile-continuous`) to record execution metrics directly into
memory-mapped profile files during test execution.

[clang-continuous]:
  https://clang.llvm.org/docs/UsersManual.html#cmdoption-fprofile-continuous

### Compiler Counter Relocation
When compiled with `-fprofile-continuous`, Clang configures the LLVM code
generator (`-mllvm -runtime-counter-relocation=true`) so that execution counter
increments reference a dynamic base pointer (`*(bias + &counter) += 1`). Each
branch and basic block counter dynamically resolves to an address within a
dedicated profile buffer mapped at program startup.

### Runtime Memory Mapping
During binary initialization, the profiling runtime (`libclang_rt.profile`)
resolves the target `.profraw` file and maps the execution counter section into
process memory using `mmap` with `MAP_SHARED`. The runtime sets the global bias
pointer to this mapped region, routing live counter increments directly into the
file-backed buffer.

### Kernel Page-Cache Synchronization
Execution counts are written directly to shared memory-mapped pages and
synchronized by the operating system kernel's page cache. Subprocesses created
via `fork()` share the same underlying memory mapping, committing statements
executed across parent and child processes directly to the profile file.

### Build System Integration
Setting `-DLIBC_ENABLE_COVERAGE=ON` in the CMake configuration passes
`-fprofile-instr-generate=libc_cov_%c%p.profraw`, `-fcoverage-mapping`, and
`-fprofile-continuous` across all LLVM-libc compilation units and test link
steps. This ensures uniform instrumentation across entrypoints and unit test
harnesses.

## Running Code Coverage Locally

### Prerequisites & Toolchain Setup

Generating coverage reports requires Clang, LLVM profile tools, CMake, and
Ninja:

* **Compiler:** Clang 18 or later (Clang 21 or later is required for MC/DC
  instrumentation).
* **LLVM Utilities:** Matching major versions of `llvm-profdata` and `llvm-cov`.
* **Build System:** CMake 3.28+ and Ninja.

#### Toolchain Discovery

If your Linux distribution packages version-suffixed binaries (e.g. `clang-21`,
`llvm-profdata-21`), discover and export them:

```bash
CLANG_MAJOR=$(clang --version | sed -n 's/.*version \([0-9]*\).*/\1/p')
export LLVM_PROFDATA=$(which llvm-profdata-$CLANG_MAJOR 2>/dev/null \
  || which llvm-profdata)
export LLVM_COV=$(which llvm-cov-$CLANG_MAJOR 2>/dev/null \
  || which llvm-cov)
```

If version-agnostic tools are directly available in your `PATH`, export:

```bash
export LLVM_PROFDATA=llvm-profdata
export LLVM_COV=llvm-cov
```

Subsequent merge and report commands reference `$LLVM_PROFDATA` and `$LLVM_COV`.

---

## Cleaning Profile Counters

Removes previously generated raw profile counter files (`.profraw`) and merged
profile databases (`.profdata`) so that new coverage runs record clean,
non-aggregated execution data:

```bash
find . -name "libc_cov_*.profraw" -delete 2>/dev/null || true
rm -f libc_full.profdata libc_mcdc.profdata libc_single.profdata \
  profraw_list.txt
```

---

## Standard Statement & Branch Coverage

Standard coverage measures physical line execution and conditional branch
outcomes across all LLVM-libc entrypoints and internal support utilities.

### 1. CMake Configuration

Configures CMake to build LLVM-libc in overlay mode, setting
`-DLIBC_ENABLE_COVERAGE=ON` to pass Clang's continuous profiling and coverage
mapping flags to the compiler:

```bash
cmake -G Ninja -S runtimes -B build-cov \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_RUNTIMES="libc" \
  -DLLVM_LIBC_FULL_BUILD=OFF \
  -DLIBC_ENABLE_COVERAGE=ON
```

### 2. Build and Execute All Unit Tests

Compiles all libc unit test executables and executes them in parallel. As each
test executes, its counters are mapped directly to disk via the OS page cache:

```bash
export LLVM_PROFILE_FILE="libc_cov_%c%p.profraw"
ninja -k 0 -C build-cov libc-unit-tests
```

:::{note}
In LLVM-libc, `libc-unit-tests` builds and executes tests in a single
invocation. The `-k 0` flag ensures Ninja continues executing all remaining
test targets even if an individual edge-case test encounters an error. To only
compile test binaries without immediately executing them, use
`ninja -C build-cov libc-unit-tests-build`.
:::

### 3. Merge Profile Counters

Scans the build tree for all generated `.profraw` files and indexes them into a
unified, sparse `.profdata` archive using `$LLVM_PROFDATA`:

```bash
find build-cov/ -name "libc_cov_*.profraw" > profraw_list.txt
"$LLVM_PROFDATA" merge -sparse -f profraw_list.txt -o libc_full.profdata
```

### 4. Generate Coverage Reports

Collects all compiled test binary paths and invokes `$LLVM_COV` to correlate
recorded profile counters against the libc source tree:

```bash
TEST_BINS=($(find build-cov -type f -executable -name "*__build__"))
OBJECT_FLAGS=()
for bin in "${TEST_BINS[@]:1}"; do
  OBJECT_FLAGS+=("-object=$bin")
done
```

Reports can be generated in different formats:

#### Option 1: Terminal Summary Report
Prints an aggregated terminal summary showing line, region, and branch coverage
percentages for each file:

```bash
"$LLVM_COV" report \
  -instr-profile=libc_full.profdata \
  "${TEST_BINS[0]}" "${OBJECT_FLAGS[@]}" \
  --show-branch-summary \
  -ignore-filename-regex=".*(test|utils).*"
```

#### Option 2: Interactive HTML Dashboard
Generates an interactive HTML dashboard containing sortable directory metrics
and syntax-highlighted source views:

```bash
"$LLVM_COV" show \
  -format=html \
  -output-dir=coverage_html \
  -instr-profile=libc_full.profdata \
  "${TEST_BINS[0]}" "${OBJECT_FLAGS[@]}" \
  --show-directory-coverage \
  --show-branches=count \
  -ignore-filename-regex=".*(test|utils).*"

# Open dashboard in browser
xdg-open coverage_html/index.html
```

---

## Modified Condition / Decision Coverage (MC/DC)

MC/DC evaluates boolean sub-conditions within compound logical expressions (such
as `if (A && B)`). It verifies that each individual sub-condition evaluates to
both true and false and independently affects the outcome of the enclosing
decision.

### 1. CMake Configuration

Configures CMake with `-fcoverage-mcdc` alongside profiling flags, enabling the
compiler frontend to generate boolean condition bitmaps for compound decisions:

```bash
cmake -G Ninja -S runtimes -B build-cov-mcdc \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_RUNTIMES="libc" \
  -DLLVM_LIBC_FULL_BUILD=OFF \
  -DLIBC_ENABLE_COVERAGE=ON \
  -DCMAKE_C_FLAGS="-fcoverage-mcdc" \
  -DCMAKE_CXX_FLAGS="-fcoverage-mcdc"
```

### 2. Build and Execute Tests

Compiles and executes all unit tests with MC/DC instrumentation enabled, saving
condition evaluation bitmasks into raw profile files upon completion:

```bash
export LLVM_PROFILE_FILE="libc_cov_%c%p.profraw"
ninja -k 0 -C build-cov-mcdc libc-unit-tests
```

### 3. Merge Profiles

Indexes and merges all MC/DC `.profraw` files into a unified
`libc_mcdc.profdata` archive for report generation:

```bash
find build-cov-mcdc/ -name "libc_cov_*.profraw" > profraw_list.txt
"$LLVM_PROFDATA" merge -sparse -f profraw_list.txt -o libc_mcdc.profdata
```

### 4. Generate Reports

Maps MC/DC bitmap records to source AST decisions and evaluates condition
independence pairs:

```bash
TEST_BINS=($(find build-cov-mcdc -type f -executable -name "*__build__"))
OBJECT_FLAGS=()
for bin in "${TEST_BINS[@]:1}"; do
  OBJECT_FLAGS+=("-object=$bin")
done
```

Reports can be generated in two formats depending on your needs:

#### Option 1: Terminal Summary Report
Displays the terminal coverage summary including MC/DC Condition and Missed
Condition percentages:

```bash
"$LLVM_COV" report \
  -instr-profile=libc_mcdc.profdata \
  "${TEST_BINS[0]}" "${OBJECT_FLAGS[@]}" \
  --show-branch-summary \
  --show-mcdc-summary \
  -ignore-filename-regex=".*(test|utils).*"
```

#### Option 2: Interactive HTML Dashboard
Produces an HTML report with expandable MC/DC decision truth tables and test
vector coverage breakdowns:

```bash
"$LLVM_COV" show \
  -format=html \
  -output-dir=coverage_mcdc_html \
  -instr-profile=libc_mcdc.profdata \
  "${TEST_BINS[0]}" "${OBJECT_FLAGS[@]}" \
  --show-directory-coverage \
  --show-branches=count \
  --show-mcdc \
  --show-mcdc-summary \
  -ignore-filename-regex=".*(test|utils).*"

# Open dashboard in browser
xdg-open coverage_mcdc_html/index.html
```

---

## Running Coverage for a Single Test

When developing or modifying a specific function, coverage can be collected for
a single test without building and executing the entire test suite.

The commands below use `libc.test.src.ctype.isalpha_test` (which tests
`libc/src/ctype/isalpha.cpp`) as an example. You can test any other entrypoint
by substituting the target name and source file path:
* **Target pattern:** `libc.test.<path_to_test>.<test_name>`
  (e.g. `libc.test.src.string.strlen_test`)
* **Source path pattern:** `libc/<path_to_source>/<source_file>.cpp`
  (e.g. `libc/src/string/strlen.cpp`)

### 1. Build and Execute the Targeted Test

Compiles and runs only the specified test binary, immediately writing execution
profile counters to disk upon completion:

```bash
export LLVM_PROFILE_FILE="libc_cov_%c%p.profraw"

# For a standard coverage build
ninja -C build-cov libc.test.src.ctype.isalpha_test

# For an MC/DC build
ninja -C build-cov-mcdc libc.test.src.ctype.isalpha_test
```

### 2. Merge the Profile

Merges the single test's raw profile into an indexed database for targeted
inspection (searching whichever build directory was compiled):

```bash
find build-cov/ build-cov-mcdc/ \
  -name "libc_cov_*.profraw" 2>/dev/null > profraw_list.txt
"$LLVM_PROFDATA" merge -sparse -f profraw_list.txt -o libc_single.profdata
```

### 3. View the Terminal Report

Reports can be viewed as an overall file summary or an annotated line-by-line
breakdown:

#### Option 1: Summary Table Report
```bash
BIN_DIR="build-cov/libc/test/src/ctype"
"$LLVM_COV" report \
  -instr-profile=libc_single.profdata \
  "$BIN_DIR/libc.test.src.ctype.isalpha_test.__build__" \
  libc/src/ctype/isalpha.cpp
```

#### Option 2: Line-by-Line & Truth Table View
```bash
BIN_DIR="build-cov-mcdc/libc/test/src/ctype"
"$LLVM_COV" show \
  -instr-profile=libc_single.profdata \
  "$BIN_DIR/libc.test.src.ctype.isalpha_test.__build__" \
  --show-branches=count \
  --show-mcdc \
  libc/src/ctype/isalpha.cpp
```

---

## Interpreting Results

For detailed documentation on the LLVM coverage reporting format, refer to the
[official Clang Source-Based Code Coverage documentation][clang-coverage].

[clang-coverage]:
  https://clang.llvm.org/docs/SourceBasedCodeCoverage.html#interpreting-reports

### Coverage Metrics Overview

* **Line Coverage:** Measures whether each physical line of executable source
  code was reached at least once during testing.
* **Branch Coverage:** Measures whether each conditional branch evaluated to
  both its `True` and `False` paths. For example, if an `if (x > 0)` branch is
  taken 10 times but never skipped, branch coverage is 50% because the `False`
  path was never exercised.
* **MC/DC Coverage:** Evaluates compound boolean expressions (such as
  `if (A && B)` or `if (A || B)`). It verifies that each individual condition
  was tested as both True and False, and demonstrated that it could
  independently change the overall outcome of the decision.

### Interpreting Reports

The summary table produced by `llvm-cov report` displays metrics across
individual source files and overall totals:

* **Regions / Missed Regions:** A region is a continuous segment of code (such
  as a function body or basic block). Missed regions indicate code blocks that
  were never executed.
* **Functions / Missed Functions:** The total number of entrypoints or
  subroutines executed vs unexecuted.
* **Lines / Missed Lines:** Physical source lines executed vs unexecuted.
* **Branches / Missed Branches:** The total count of decision directions (both
  True and False) evaluated.
* **MC/DC Conditions / Missed Conditions:** The count of individual boolean
  sub-conditions that demonstrated independent decision control.

### Interpreting MC/DC Truth Tables

When inspecting with `--show-mcdc`, `llvm-cov` displays an MC/DC analysis table
beneath each compound decision. For instance, consider the following decision:

```text
   19|  if (c < 0 || c > 255)
  -----------------------------------------------
  | Conditions: C1 = (c < 0), C2 = (c > 255)
  |
  | Executed Test Vectors:
  |    C1, C2    Result
  | 1 { F,  F  = F      }  (tested with c = 'a')
  | 2 { T,  -  = T      }  (tested with c = -1)
  |
  | C1-Pair: covered (1, 2)
  | C2-Pair: not covered
  | MC/DC Coverage: 50.00%
  -----------------------------------------------
```

* **Conditions:** **C1** represents `c < 0` and **C2** represents `c > 255`.
* **Executed Vectors:**
  * **Vector 1 (`F, F = F`):** Tested with a valid character (`c = 'a'`). Both
    C1 and C2 evaluated False, producing an overall False result.
  * **Vector 2 (`T, - = T`):** Tested with a negative value (`c = -1`). C1
    evaluated True, producing an overall True result. The hyphen (`-`)
    indicates C2 was short-circuited and not evaluated.
* **Condition Pairs:**
  * **`C1-Pair: covered (1, 2)`:** Comparing Vector 1 and Vector 2 proves that
    changing C1 from False to True directly flipped the result from False to
    True. C1 is fully covered.
  * **`C2-Pair: not covered`:** C2 was never tested in a state where it
    independently turned the result True while C1 was False.
* **Reaching 100% Coverage:** Add a test with a value above 255 (`c = 256`).
  This executes Vector 3 (`F, T = T`), forming the independence pair `(1, 3)`
  for C2 and reaching 100% MC/DC coverage.
