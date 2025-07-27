# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System and Common Commands

LLVM uses CMake as its build system. The typical workflow involves:

**Initial Configuration:**
```bash
# Configure with Ninja (most developers use this)
cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON

# Build specific projects (example with Clang and LLD)
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_BUILD_TYPE=Release
```

**Building:**
```bash
# Build everything
cmake --build build
# Or using ninja directly
ninja -C build

# Build specific targets
ninja -C build llvm-opt clang
```

**Running Tests:**
```bash
# Run all LLVM regression tests
ninja -C build check-llvm

# Run all unit tests
ninja -C build check-llvm-unit

# Run specific project tests
ninja -C build check-clang
ninja -C build check-lld

# Run all tests (unit + regression)
ninja -C build check-all

# Run individual tests with lit
llvm-lit llvm/test/Analysis/BasicAA/
llvm-lit path/to/specific/test.ll
```

**Key CMake Options:**
- `LLVM_ENABLE_PROJECTS`: Semicolon-separated list of projects to build (clang, lld, lldb, etc.)
- `LLVM_TARGETS_TO_BUILD`: Target architectures to build (default: "all")
- `CMAKE_BUILD_TYPE`: Debug, Release, RelWithDebInfo, MinSizeRel
- `LLVM_ENABLE_ASSERTIONS`: ON/OFF (default ON for Debug builds)
- `LLVM_USE_LINKER=lld`: Use LLD linker for faster linking

## Architecture Overview

**Major Components:**
- **llvm/**: Core LLVM infrastructure (IR, passes, code generation, tools)
- **clang/**: C/C++/Objective-C compiler frontend
- **clang-tools-extra/**: Additional Clang-based tools (clang-tidy, clangd, etc.)
- **lld/**: LLVM linker
- **lldb/**: LLVM debugger
- **compiler-rt/**: Runtime libraries (sanitizers, profiling, etc.)
- **libcxx/**: C++ standard library implementation
- **libcxxabi/**: C++ ABI library
- **polly/**: Polyhedral optimizer
- **mlir/**: Multi-Level IR framework
- **flang/**: Fortran frontend
- **bolt/**: Binary optimization and layout tool

**LLVM Core Structure:**
- **lib/**: Core implementation (Analysis, Transforms, CodeGen, etc.)
- **include/llvm/**: Public headers
- **tools/**: Command-line tools (opt, llc, llvm-dis, etc.)
- **test/**: Regression tests organized by component
- **unittests/**: Unit tests using Google Test

**Key Directories in llvm/lib/:**
- `Analysis/`: Various analysis passes
- `Transforms/`: Optimization passes (IPO, Scalar, Vectorize, etc.)
- `CodeGen/`: Code generation framework and target-specific backends
- `Target/`: Architecture-specific backends (X86, AArch64, ARM, etc.)
- `IR/`: LLVM IR core classes and utilities
- `Support/`: General utilities and data structures

## Testing Infrastructure

**Test Organization:**
- **Regression tests**: `llvm/test/` - FileCheck-based tests for specific functionality
- **Unit tests**: `llvm/unittests/` - Google Test-based tests for individual components
- **Integration tests**: Use lit (LLVM Integrated Tester) framework

**Test Types:**
- `.ll` files: LLVM IR tests
- `.c/.cpp` files: Source-level tests (usually for Clang)
- Unit tests: C++ tests using Google Test framework

**Running Specific Tests:**
```bash
# Run tests for a specific pass/analysis
llvm-lit llvm/test/Analysis/LoopInfo/
llvm-lit llvm/test/Transforms/InstCombine/

# Run with specific flags
llvm-lit -v llvm/test/path/to/test.ll
llvm-lit --filter="NamePattern" test/directory/
```

## Development Workflow

**Common Patterns:**
- Most code follows LLVM coding standards (CamelCase for types, lowerCamelCase for variables)
- Use `LLVM_DEBUG()` macro for debug output
- Prefer LLVM data structures (SmallVector, DenseMap, etc.) over STL equivalents
- Use LLVM's error handling mechanisms (Expected<T>, Error, etc.)

**Pass Development:**
- Analysis passes: Inherit from AnalysisInfoMixin or AnalysisPass
- Transformation passes: Inherit from PassInfoMixin or FunctionPass/ModulePass
- Register passes in the appropriate PassRegistry

**Adding Tests:**
- Add regression tests to appropriate subdirectory in `llvm/test/`
- Use FileCheck for output verification
- Add unit tests to `llvm/unittests/` for complex logic
- Tests should be minimal and focused on specific functionality

**Documentation:**
- Core docs in `llvm/docs/` (ReStructuredText format)
- Doxygen comments for public APIs
- Use `ninja docs-llvm-html` to build documentation

## File Locations for Common Tasks

**Analysis Pass Development:**
- Headers: `llvm/include/llvm/Analysis/`
- Implementation: `llvm/lib/Analysis/`
- Tests: `llvm/test/Analysis/PassName/`
- Unit tests: `llvm/unittests/Analysis/`

**Transform Pass Development:**
- Headers: `llvm/include/llvm/Transforms/`
- Implementation: `llvm/lib/Transforms/`
- Tests: `llvm/test/Transforms/PassName/`

**Target-Specific Code:**
- Headers: `llvm/include/llvm/Target/TargetName/`
- Implementation: `llvm/lib/Target/TargetName/`
- Tests: `llvm/test/CodeGen/TargetName/`

**Tools:**
- Implementation: `llvm/tools/ToolName/`
- Tests: `llvm/test/tools/ToolName/`

## LLDB-Specific Development

**LLDB Structure:**
- **lldb/source/**: Core LLDB implementation
- **lldb/include/**: Public LLDB headers
- **lldb/test/**: Test suites (API, Shell tests)
- **lldb/tools/**: LLDB tools and utilities
- **lldb/unittests/**: Unit tests for LLDB components

**LLDB Plugin Development:**
- **Language plugins**: `lldb/source/Plugins/Language/LanguageName/`
- **Type systems**: `lldb/source/Plugins/TypeSystem/`
- **Symbol files**: `lldb/source/Plugins/SymbolFile/`
- **Expression parsers**: `lldb/source/Plugins/ExpressionParser/`

**LLDB Build Configuration for Development:**
```bash
# Recommended LLDB build for language plugin development
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;lldb;flang" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLDB_INCLUDE_TESTS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld

# For debugging LLDB itself, use Debug instead of RelWithDebInfo
# cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug ...
```

**LLDB Testing:**
```bash
# Run all LLDB tests
ninja -C build check-lldb

# Run LLDB API tests
ninja -C build check-lldb-api

# Run LLDB unit tests
ninja -C build check-lldb-unit

# Run specific LLDB test
llvm-lit lldb/test/API/lang/c/
lldb-dotest lldb/test/API/functionalities/breakpoint/
```

**Language Plugin Development Pattern:**
1. Create plugin directory: `lldb/source/Plugins/Language/LanguageName/`
2. Implement `LanguageName.h` and `LanguageName.cpp` inheriting from `Language`
3. Use `LLDB_PLUGIN_DEFINE(LanguageName)` macro for plugin definition
4. Register with PluginManager in Initialize/Terminate methods
5. Add to `lldb/source/Plugins/Language/CMakeLists.txt` using `add_subdirectory()`
6. Create tests in `lldb/test/API/lang/languagename/` with proper Python test classes
7. Follow LLVM coding standards: C++17, CamelCase types, lowerCamelCase variables

**Key LLDB Classes for Language Plugins:**
- `Language`: Base class for language-specific functionality
- `TypeSystem`: Handles type representation and operations  
- `FormatManager`: Manages data formatters for types
- `ExpressionParser`: Parses and evaluates expressions
- `DWARFASTParser`: Parses DWARF debug information into types

**Fortran Development Workflow:**
- Use TDD: Write failing test first, implement to pass, refactor
- Build only LLDB components: `ninja -C build lldb lldb-test`
- Test frequently: `llvm-lit lldb/test/API/lang/fortran/`
- Format code: `git clang-format HEAD~1` before committing
- Follow LLVM conventions: single commit per PR, no unrelated changes
- Use selective building to save time on large codebase

**LLVM Contribution Guidelines:**
- Base patches on recent `main` branch commit
- Include test cases with all patches
- Format code with clang-format
- Follow C++17 standards and LLVM data structures (SmallVector, DenseMap)
- Use LLVM error handling (Expected<T>, Error) over exceptions