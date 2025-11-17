# Customizable Functions Feature Development Guide

This guide covers developing and testing the Customizable Functions feature in Clang.

## Quick Start

### First Time Setup

```bash
# 1. Configure the build (debug mode, optimized for development)
./custom-functions-dev.sh configure debug

# 2. Build clang
./custom-functions-dev.sh build

# 3. Verify the build
./custom-functions-dev.sh info
```

### Development Workflow

```bash
# Make changes to source files...

# Build only what changed
./custom-functions-dev.sh build

# Run customizable functions tests
./custom-functions-dev.sh test customizable

# Or run specific test categories
./custom-functions-dev.sh test parser
./custom-functions-dev.sh test sema
./custom-functions-dev.sh test codegen
```

## Build Script Commands

### Configure

```bash
# Debug build (default - best for development)
./custom-functions-dev.sh configure debug

# Release build (for performance testing)
./custom-functions-dev.sh configure release

# Minimal build (fastest iteration)
./custom-functions-dev.sh configure minimal
```

### Build

```bash
# Build clang
./custom-functions-dev.sh build

# Build specific target
./custom-functions-dev.sh build check-clang-sema

# Build with custom job count
BUILD_JOBS=8 ./custom-functions-dev.sh build
```

### Test

```bash
# Run all customizable functions tests
./custom-functions-dev.sh test customizable

# Run parser tests
./custom-functions-dev.sh test parser

# Run semantic analysis tests
./custom-functions-dev.sh test sema

# Run code generation tests
./custom-functions-dev.sh test codegen

# Run AST tests
./custom-functions-dev.sh test ast

# Run all Clang tests
./custom-functions-dev.sh test all

# Run tests matching specific pattern
./custom-functions-dev.sh test "custom.*syntax"
```

### Utility Commands

```bash
# Show build information
./custom-functions-dev.sh info

# Clean build directory
./custom-functions-dev.sh clean

# Rebuild from scratch
./custom-functions-dev.sh rebuild

# Show help
./custom-functions-dev.sh help
```

## Build Optimization

The build script uses several optimizations for faster development:

- **Ninja build system**: Parallel builds with dependency tracking
- **Optimized TableGen**: Faster build times
- **Split DWARF**: Faster linking with debug info
- **Single target (X86)**: Reduces build time
- **Minimal LLVM projects**: Only builds Clang, not other tools

### Build Times (Approximate)

On a typical development machine (8 cores):

- **Initial build**: 20-40 minutes (full Clang build)
- **Incremental parser change**: 30-60 seconds
- **Incremental Sema change**: 1-2 minutes
- **Incremental test-only**: 5-10 seconds

## Development Tips

### Fast Iteration Cycle

1. **Use incremental builds**: The script automatically detects changes
2. **Test specific categories**: Don't run all tests every time
3. **Use minimal build mode**: For rapid prototyping

```bash
# Example fast iteration
vim clang/lib/Parse/ParseDecl.cpp
./custom-functions-dev.sh build
./custom-functions-dev.sh test parser
```

### Debugging Build Issues

```bash
# Show detailed build information
./custom-functions-dev.sh info

# Clean and rebuild
./custom-functions-dev.sh rebuild debug

# Verbose build output
cd build-custom-functions && ninja -v clang
```

### Running Individual Tests

```bash
# Build directory contains lit tool
cd build-custom-functions

# Run a specific test file
./bin/llvm-lit -v ../clang/test/Parser/cxx-customizable-functions.cpp

# Run with verbose output
./bin/llvm-lit -v -a ../clang/test/SemaCXX/customizable-functions-*.cpp
```

### Using the Built Clang

```bash
# Direct path to built clang
./build-custom-functions/bin/clang --version

# Test the custom keyword (when implemented)
./build-custom-functions/bin/clang -std=c++20 -fcustomizable-functions -fsyntax-only test.cpp

# See generated AST
./build-custom-functions/bin/clang -std=c++20 -fcustomizable-functions -Xclang -ast-dump test.cpp
```

## Project Structure

```
llvm-project/
├── custom-functions-dev.sh    # Main build script
├── CUSTOMIZABLE_FUNCTIONS_DEV.md      # This file
├── build-custom-functions/            # Build output directory
│   ├── bin/clang                      # Built clang binary
│   └── compile_commands.json          # For IDE integration
├── clang/
│   ├── docs/
│   │   ├── CustomizableFunctionsDesign.md     # Design document
│   │   └── CustomizableFunctionsTestPlan.md   # Test plan
│   ├── include/clang/
│   │   ├── Basic/
│   │   │   ├── TokenKinds.def         # Add 'custom' keyword here
│   │   │   └── LangOptions.def        # Add language option here
│   │   ├── Parse/
│   │   │   └── Parser.h               # Parser interface
│   │   ├── Sema/
│   │   │   ├── Sema.h                 # Semantic analysis interface
│   │   │   └── SemaCustomFunction.h   # Custom function transformation (new)
│   │   └── AST/
│   │       └── Decl.h                 # AST node declarations
│   ├── lib/
│   │   ├── Parse/
│   │   │   └── ParseDecl.cpp          # Parse 'custom' keyword
│   │   ├── Sema/
│   │   │   ├── SemaDecl.cpp           # Semantic analysis
│   │   │   └── SemaCustomFunction.cpp # Transform logic (new)
│   │   └── AST/
│   │       └── Decl.cpp               # AST implementation
│   └── test/
│       ├── Parser/
│       │   └── cxx-customizable-functions-*.cpp
│       ├── SemaCXX/
│       │   └── customizable-functions-*.cpp
│       ├── CodeGenCXX/
│       │   └── customizable-functions-*.cpp
│       └── AST/
│           └── customizable-functions-*.cpp
```

## Common Development Tasks

### Adding a New Keyword

1. Add to `clang/include/clang/Basic/TokenKinds.def`
2. Rebuild: `./custom-functions-dev.sh build`
3. Test: `./custom-functions-dev.sh test parser`

### Adding Parser Support

1. Modify `clang/lib/Parse/ParseDecl.cpp`
2. Add tests in `clang/test/Parser/`
3. Build and test:
   ```bash
   ./custom-functions-dev.sh build
   ./custom-functions-dev.sh test parser
   ```

### Adding Semantic Analysis

1. Create `clang/lib/Sema/SemaCustomFunction.cpp`
2. Add hook in `clang/lib/Sema/SemaDecl.cpp`
3. Add tests in `clang/test/SemaCXX/`
4. Build and test:
   ```bash
   ./custom-functions-dev.sh build
   ./custom-functions-dev.sh test sema
   ```

### Adding Code Generation

1. Modify `clang/lib/CodeGen/CGDecl.cpp`
2. Add tests in `clang/test/CodeGenCXX/`
3. Build and test:
   ```bash
   ./custom-functions-dev.sh build
   ./custom-functions-dev.sh test codegen
   ```

## IDE Integration

### Compile Commands (for clangd, CLion, etc.)

The build script automatically creates a symlink to `compile_commands.json`:

```bash
llvm-project/compile_commands.json -> build-custom-functions/compile_commands.json
```

This enables IDE features like:
- Code completion
- Jump to definition
- Error highlighting
- Refactoring tools

### VS Code

Install the clangd extension and it will automatically find the compile commands.

### CLion

CLion will detect the CMake project automatically. Point it to `build-custom-functions/`.

## Testing Strategy

### Test Categories

1. **Parser Tests**: Syntax validation
   - `./custom-functions-dev.sh test parser`

2. **Semantic Tests**: Type checking, constraints
   - `./custom-functions-dev.sh test sema`

3. **CodeGen Tests**: LLVM IR generation
   - `./custom-functions-dev.sh test codegen`

4. **AST Tests**: AST structure verification
   - `./custom-functions-dev.sh test ast`

5. **Integration Tests**: End-to-end workflows
   - `./custom-functions-dev.sh test customizable`

### Writing New Tests

Create test files following this pattern:

```cpp
// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// Test case description
custom void my_test() { }  // expected-no-diagnostics

// Or with expected errors
custom void bad_test();  // expected-error {{custom functions must have a body}}
```

Place in appropriate directory:
- Parser tests: `clang/test/Parser/`
- Semantic tests: `clang/test/SemaCXX/`
- CodeGen tests: `clang/test/CodeGenCXX/`

## Troubleshooting

### Build Fails

```bash
# Clean and rebuild
./custom-functions-dev.sh rebuild

# Check for CMake issues
cd build-custom-functions
cmake ..

# Verbose build to see errors
ninja -v clang
```

### Tests Fail

```bash
# Run specific test with verbose output
cd build-custom-functions
./bin/llvm-lit -v ../clang/test/Parser/cxx-customizable-functions.cpp -a

# Check test expectations
cat ../clang/test/Parser/cxx-customizable-functions.cpp
```

### Performance Issues

```bash
# Use release build for performance testing
./custom-functions-dev.sh configure release
./custom-functions-dev.sh build

# Increase parallel jobs (if you have RAM)
BUILD_JOBS=16 ./custom-functions-dev.sh build
```

## Environment Variables

- `BUILD_JOBS`: Number of parallel build jobs (default: nproc)
- `BUILD_TYPE`: Override build type (Debug/Release)
- `CC`: C compiler to use
- `CXX`: C++ compiler to use

Example:
```bash
export BUILD_JOBS=12
export CC=clang
export CXX=clang++
./custom-functions-dev.sh configure debug
```

## References

- [Design Document](clang/docs/CustomizableFunctionsDesign.md)
- [Test Plan](clang/docs/CustomizableFunctionsTestPlan.md)
- [LLVM Testing Infrastructure](https://llvm.org/docs/TestingGuide.html)
- [Clang Internals Manual](https://clang.llvm.org/docs/InternalsManual.html)

## Next Steps

Once the build is configured:

1. Review the design documents
2. Start with Phase 1: Add keyword and basic parsing
3. Add tests incrementally
4. Iterate quickly with the build script

Happy coding!
