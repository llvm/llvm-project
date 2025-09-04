// This test checks retrieving dwo file diercetly with dwo name,
// if the Debug Compilation Dir does not exist.

int main() { return 0; }

// RUN: rm -rf %t && mkdir -p %t && cd %t
// RUN: %clang %cflags -g -gsplit-dwarf -fdebug-compilation-dir=/path/does/not/exist %s -o main.exe
// RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -update-debug-sections  2>&1 | FileCheck %s -check-prefix=NOT-EXIST

// NOT-EXIST: BOLT-WARNING: Debug Fission: Debug Compilation Dir wrong for

// RUN: rm -rf %t && mkdir -p %t && cd %t
// RUN: %clang %cflags -g -gsplit-dwarf -fdebug-compilation-dir=/path/does/not/exist %s -o %t/main.exe
// RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -update-debug-sections  2>&1 | FileCheck %s -check-prefix=FOUND

// FOUND-NOT: Debug Fission: DWO debug information for
