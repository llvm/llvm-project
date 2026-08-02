# LLDB Debugger: RISC-V (RV64GC) and x86 Build & Debugging Guide

This comprehensive guide outlines the procedure for configuring and compiling the LLDB debugger from source to support both x86 and RISC-V 64-bit (`rv64gc`) architectures. Furthermore, it details the end-to-end workflow for debugging bare-metal RISC-V 64 code utilizing the Spike simulator, OpenOCD, and LLDB.

---

## 1. Build Modes

LLVM and LLDB employ CMake for their build system. The chosen build mode determines the optimization level and the inclusion of debug symbols. 

You specify the build mode using the `-DCMAKE_BUILD_TYPE=<mode>` flag during the CMake configuration step (see Section 2.1).

The available build modes are:

*   **Release (Recommended):** `-DCMAKE_BUILD_TYPE=Release`
    Builds with full optimization and strips debug symbols. This yields the fastest execution time, the smallest binary size, and the best runtime performance for the debugger.
*   **Debug:** `-DCMAKE_BUILD_TYPE=Debug`
    Builds with no optimizations and includes comprehensive debug symbols. Use this *only* if you are actively modifying and debugging the LLDB C++ source code itself. Note that this mode produces a significantly larger binary, runs slower, and uses a lot of disk space.
*   **RelWithDebInfo:** `-DCMAKE_BUILD_TYPE=RelWithDebInfo`
    Optimized for performance but includes debug symbols. Ideal for generating stack traces upon crashes without a severe performance penalty.
*   **MinSizeRel:** `-DCMAKE_BUILD_TYPE=MinSizeRel`
    Optimized strictly for minimal binary footprint rather than execution speed.

---

## 2. Compiling LLDB

Follow these steps to configure CMake to support both X86 and RISC-V (including `riscv64gc`) target architectures, and compile the project using Ninja.

### Step 2.1: Configure CMake
You must execute the `cmake` command from within a dedicated build directory. You only need to run this command once to set up the build targets and generators.

1. Navigate to the build directory:
   ```bash
   cd /home/saif/Desktop/Si-Vision/Si-Vision-internship/LLDB_SorceCode/llvm-project/build
   ```
2. Run the CMake configuration command:
   ```bash
   cmake -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_PROJECTS="clang;lldb" \
     -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
     -DLLVM_LINK_LLVM_DYLIB=ON \
     -DLLVM_OPTIMIZED_TABLEGEN=ON \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
     ../llvm
   ```

### Step 2.2: Build with Ninja
After configuring, or after editing LLDB's C++ source code files (located under `llvm-project/lldb/`), you will compile the project using Ninja.

Execute the following command from your build directory:
```bash
ninja -j 4 lldb
```

> **Tip:** For daily code edits, you only need to run this `ninja` command to build your changes. The `-j 4` flag restricts the compilation to 4 concurrent jobs to prevent memory overload and system lag.

---

## 3. RISC-V 64 Debugging Guide (rv64gc)

This section explains how to build, run, and debug bare-metal RISC-V 64 code (`main.elf`) using the Spike simulator, OpenOCD, and LLDB.

> **Important:** Make sure all three of the following commands are run in **separate terminal sessions** in the exact sequence shown below.

### Step 3.1: Launch the Spike Simulator
Start the Spike simulator, enabling the JTAG Remote Bitbang interface (port 9824) and halting the core on start to wait for the debugger connection:

```bash
spike --isa=rv64gc --rbb-port=9824 --halted main.elf
```
*(Alternatively, you can run: `make spike-debug` inside the `code_testing/Baremetal_code_test/` directory).*

### Step 3.2: Launch OpenOCD
Start OpenOCD to bridge the JTAG Remote Bitbang connection from Spike (port 9824) to a GDB Remote Serial Protocol (RSP) server (port 3333):

```bash
openocd -c "adapter driver remote_bitbang; remote_bitbang_port 9824; remote_bitbang_host localhost; jtag newtap riscv cpu -irlen 5 -expected-id 0xdeadbeef; target create riscv.cpu riscv -endian little -coreid 0 -chain-position riscv.cpu; gdb_port 3333"
```
*(Alternatively, you can run: `make openocd` inside the `code_testing/Baremetal_code_test/` directory).*

### Step 3.3: Launch LLDB & Connect
Start LLDB, configure it for `riscv64`, load the target symbols from your ELF file, and connect to the OpenOCD RSP server:

```bash
/home/saif/Desktop/Si-Vision/Si-Vision-internship/LLDB_SorceCode/llvm-project/build_multitarget/bin/lldb --arch riscv64 main.elf -o "gdb-remote 3333"
```
*(Alternatively, you can run: `make lldb` inside the `code_testing/Baremetal_code_test/` directory).*
