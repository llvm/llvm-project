(code_coverage)=

# Source-Level Code Coverage

LLVM-libc supports source-level code coverage for its unit tests.

Because `llvm-libc` unit tests are built as freestanding binaries (`-nostdlib`), standard `compiler-rt` coverage workflows fail in this environment as they inherently rely on the host's standard C library (`fopen`, `fwrite`), which is intentionally omitted to prevent host contamination.

To bypass this constraint, the `LibcTestMain.cpp` test harness implements a custom profiling dumper:

1. It silences the default `compiler-rt` dumper by overriding the global symbol: `extern "C" char __llvm_profile_filename[] = "/dev/null";`.
2. It hooks into `atexit()` to dump coverage before the process ends. *(Note: Death tests that terminate via `_exit()`, `abort()`, or unhandled signals inherently bypass this hook. The parent test runner still correctly dumps its overall profile).*
3. It determines the required buffer size via `__llvm_profile_get_size_for_buffer()` and allocates memory using `LIBC_NAMESPACE::linux_syscalls::mmap`.
4. It extracts the raw profiling data from the compiler into the memory segment using `__llvm_profile_write_buffer()`.
5. It writes the segment to a `.profraw` file using internal Linux syscall wrappers (`open`, `write`, `close`, `munmap`).

## Limitations

- **OS Support:** Because this relies on Linux system call wrappers, coverage extraction is strictly gated behind `#if defined(__linux__)`. On macOS or Windows builds, the coverage dumping step is gracefully bypassed, allowing the tests to compile normally.

## 1. Setup and Configuration

Before running any tests, you must clear old profile data and configure your CMake build directory to generate coverage instrumentation.

```bash
# 1. Clear previous profile artifacts
find . -name "libc_cov_*.profraw" -delete
rm -f libc_full.profdata profraw_list.txt

# 2. Configure the build directory
cmake -G Ninja -S runtimes -B build-cov \
  -DLLVM_ENABLE_RUNTIMES="libc" \
  -DLLVM_LIBC_FULL_BUILD=ON \
  -DLLVM_LIBC_ENABLE_COVERAGE=ON \
  -DCMAKE_CXX_COMPILER=clang++-19 \
  -DCMAKE_C_COMPILER=clang-19
```

## 2. Build and Run Tests

You have three options depending on how much of the library you want to test. (Targeted testing is significantly faster for local debugging).

Test targets in `llvm-libc` follow a strict naming convention based on their directory structure: `libc.test.<directory_path>.<test_name>.__unit__`.

### Option A: Whole Codebase
To run the entire unit test suite (approx. 1,776 tests) and generate a massive, whole-codebase report:

```bash
ninja -C build-cov check-libc

# If using Option A, do not define FILES_TO_REPORT in Step 3.
```

### Option B: Single Target
To instantly run coverage on a single test, specify its exact Ninja target. For example, to test `isalpha`:

```bash
# Run only the specific target
ninja -C build-cov libc.test.src.ctype.isalpha_test.__unit__

# Define the source file to filter the report in Step 3
FILES_TO_REPORT="libc/src/ctype/isalpha.cpp"
```

### Option C: Multiple Targets
To run multiple isolated tests simultaneously, pass them as a space-separated list. For example, testing `isalpha` and `isdigit`:

```bash
# Define your targets 
TARGETS="libc.test.src.ctype.isalpha_test.__unit__ libc.test.src.ctype.isdigit_test.__unit__"

# Run the targets
ninja -C build-cov $TARGETS

# Define the source files to filter the report in Step 3
FILES_TO_REPORT="libc/src/ctype/isalpha.cpp libc/src/ctype/isdigit.cpp"
```

## 3. Generate the Report

Once your tests have finished running, merge the raw profile data and extract the executables to map the coverage back to the source code.

```bash
# 1. Merge raw profiles
find . -name "libc_cov_*.profraw" > profraw_list.txt
llvm-profdata-19 merge -sparse --input-files=profraw_list.txt -o libc_full.profdata

# 2. Extract executables
EXECUTABLES=($(find build-cov -type f -executable -name "*__build__"))
OBJECTS=("${EXECUTABLES[@]:1}")
OBJECTS=("${OBJECTS[@]/#/-object=}")
```

### Choose Your Output Type

You can specify the format of your final coverage report by changing the output command.

**Option A: Terminal Summary Table (Text)**
This provides a quick text-based summary of your coverage percentages directly in the terminal:
```bash
llvm-cov-19 report -instr-profile=libc_full.profdata "${EXECUTABLES[0]}" "${OBJECTS[@]}" $FILES_TO_REPORT
```

**Option B: Line-by-Line Interactive Webpage (HTML)**
This generates an interactive HTML website so you can visually inspect exactly which lines of code are missing coverage. You can change the `OUTPUT_DIR` variable to save it wherever you prefer:
```bash
OUTPUT_DIR="coverage_html"
llvm-cov-19 show -instr-profile=libc_full.profdata -format=html -output-dir=$OUTPUT_DIR "${EXECUTABLES[0]}" "${OBJECTS[@]}" $FILES_TO_REPORT
```
*(After running this, open `coverage_html/index.html` in your web browser).*
