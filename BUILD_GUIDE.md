# LLDB & LLVM Build, Architecture & Ninja Guide

This guide explains:
1. Why **Clang** and **LLVM** are required to build and run **LLDB**.
2. How to build **LLDB** efficiently (Fast Developer Build vs Full System Build).
3. Explanation of **CMake Configuration Flags**.
4. How to **Reconfigure CMake / Regenerate Ninja Build Files** with new parameters.
5. How to use **Ninja** with various command-line flags.

---

## 1. Why `clang/` and `llvm/` are Required for LLDB

LLDB is not a standalone executable—it is built on top of LLVM and Clang libraries:

### Why `clang/` is required:
* **The Expression Parser Engine**: When you type `p (int)myVar + 5` or `expr myVector.size()` inside LLDB, LLDB compiles that code snippet on the fly.
* LLDB embeds Clang's **AST (Abstract Syntax Tree)**, **Lexer**, **Parser**, and **CodeGen** to evaluate C, C++, and Objective-C expressions accurately. Without Clang, LLDB would not understand C++ syntax, namespaces, templates, or types.

### Why `llvm/` is required:
* **Binary & Symbol Parsing**: LLDB relies on LLVM libraries (`LLVMObject`, `LLVMDebugInfoDWARF`, `LLVMSymbolize`) to read ELF/DWARF executables, extract symbols, and read debug info.
* **Disassembly & Target Architectures**: LLVM provides instruction disassemblers for x86_64, ARM, RISC-V, etc.
* **Core Utilities**: LLVM supplies core data structures, memory management, and process handling (`LLVMSupport`).

---

## 2. Fast & Lightweight Developer Build (Recommended)

Reduces compilation from **5,630+ steps down to ~1,500 steps** for fast iteration when modifying LLDB source code.

```bash
# Navigate to llvm-project root and create build directory
cd /home/saif/Desktop/Si-Vision/Si-Vision-internship/LLDB_SorceCode/llvm-project
mkdir -p build && cd build
rm -rf *

# Reconfigure CMake with lightweight developer flags
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lldb" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  ../llvm

# Build LLDB using 4 CPU cores (prevents high RAM usage)
ninja -j 4 lldb
```

Run your custom LLDB binary:
```bash
./bin/lldb --version
./bin/lldb
```

---

## 3. How to Reconfigure CMake / Regenerate `build.ninja` with New Parameters

### When do you need to regenerate `build.ninja`?
* **NO** when making normal edits to C++ code (`.cpp` or `.h`). Just run `ninja -j 4 lldb`.
* **YES** when you want to change global build settings (e.g., switching from `Release` to `Debug`, adding `lld`, or changing target CPU architectures).

---

### Step-by-Step Reconfiguration Methods:

#### Method A: Update Existing Build (In-Place Reconfiguration)
Keeps compiled object files so you don't rebuild from scratch, but updates `build.ninja` with your new parameter:

```bash
cd /home/saif/Desktop/Si-Vision/Si-Vision-internship/LLDB_SorceCode/llvm-project/build

# Example: Change build type to Debug and add LLD linker project
cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS="clang;lldb;lld" ../llvm

# Recompile with Ninja
ninja -j 4 lldb
```

#### Method B: Clean Reconfiguration (Fresh Start)
Wipes the old cache completely and generates a brand-new `build.ninja`:

```bash
cd /home/saif/Desktop/Si-Vision/Si-Vision-internship/LLDB_SorceCode/llvm-project/build
rm -rf *

cmake -G Ninja <YOUR_NEW_FLAGS> ../llvm
ninja -j 4 lldb
```

---

## 4. CMake Configuration Options & Parameter Reference

| Parameter / Flag | Options | Description & Usage |
| :--- | :--- | :--- |
| **`-DCMAKE_BUILD_TYPE`** | `Release`<br>`Debug`<br>`RelWithDebInfo` | **`Release`**: Fastest compilation, no heavy debug symbols.<br>**`Debug`**: Adds full `-g` debug symbols so you can step-debug LLDB itself using gdb/lldb.<br>**`RelWithDebInfo`**: Optimized code with debug symbols. |
| **`-DLLVM_ENABLE_PROJECTS`** | `"clang;lldb"`<br>`"clang;lldb;lld"`<br>`"clang;lldb;compiler-rt"` | Specifies sub-projects to include in the build. Add `lld` to build LLVM's linker, or `compiler-rt` for runtime sanitizers. |
| **`-DLLVM_TARGETS_TO_BUILD`** | `"X86"`<br>`"AArch64"`<br>`"RISCV"`<br>`"all"` | **`"X86"`**: Builds target generators only for 64-bit Intel/AMD CPUs (fastest).<br>**`"all"`**: Builds target generators for 20+ architectures (ARM, MIPS, PowerPC, RISC-V, etc.). |
| **`-DLLVM_LINK_LLVM_DYLIB`** | `ON` / `OFF` | **`ON`**: Compiles LLVM into a single shared dynamic library (`libLLVM.so`), speeding up linking by 10x.<br>**`OFF`**: Statically links object libraries into binaries. |
| **`-DLLVM_OPTIMIZED_TABLEGEN`** | `ON` / `OFF` | Compiles the internal code generator tool (`llvm-tblgen`) with optimization for fast tablegen parsing. |
| **`-DLLVM_ENABLE_ASSERTIONS`** | `ON` / `OFF` | Enables internal LLVM `assert()` sanity checks. Useful when developing new LLDB features (`-DLLVM_ENABLE_ASSERTIONS=ON`). |
| **`-DCMAKE_EXPORT_COMPILE_COMMANDS`** | `ON` / `OFF` | Generates `compile_commands.json` in `build/` for VS Code, clangd, or C++ IDE autocompletion. |

---

## 5. Ninja Build Commands & Flags Guide

Once CMake generates `build.ninja` inside the `build/` directory, you interact directly with `ninja`.

### Essential Ninja Build Commands:

```bash
# 1. Build only LLDB and its dependencies using 4 CPU cores (Recommended)
ninja -j 4 lldb

# 2. Build only the LLDB C++ library (liblldb.so)
ninja -j 4 liblldb

# 3. Build everything in the current CMake configuration
ninja -j 4

# 4. Limit parallel CPU jobs (e.g. use 4 CPU cores to avoid high RAM usage)
ninja -j 4 lldb

# 5. Verbose output (shows exact compiler commands, -I flags, -D defines)
ninja -v lldb

# 6. List all available build targets generated by CMake
ninja -t targets

# 7. Clean compiled object files and binaries
ninja clean

# 8. Run LLDB unit tests and regression suite
ninja check-lldb
```

### Ninja Flags Summary Table:

| Ninja Command / Flag | Function | Use Case |
| :--- | :--- | :--- |
| **`ninja <target>`** | Builds specified target (e.g. `lldb`, `clang`, `liblldb`). | Building specific tool without building everything. |
| **`ninja -j N`** | Sets maximum concurrent jobs to `N` (e.g. `ninja -j4`). | **Crucial**: Prevents system freeze / out-of-memory errors on laptops. |
| **`ninja -v`** | Verbose mode: prints full `g++`/`clang++` command lines. | Debugging compiler flags or missing header errors. |
| **`ninja -t targets`** | Queries Ninja tool to list all valid build targets. | Finding exact target names (e.g. `lldb-server`). |
| **`ninja clean`** | Removes generated object `.o` files and binaries. | Performing a fresh recompilation. |
| **`ninja check-lldb`** | Runs test suite for LLDB. | Verifying changes don't break existing features. |

---

## 6. Upgrading CMake (Optional)

LLVM 24+ recommends **CMake 3.31.0 or newer**. Upgrade via `pip`:

```bash
pip install --upgrade cmake
cmake --version
```
