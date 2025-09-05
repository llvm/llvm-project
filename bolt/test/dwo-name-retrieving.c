// This test checks retrieving dwo file diercetly with dwo name,
// if the Debug Compilation Dir does not exist.

int main() { return 0; }

// RUN: rm -rf %t && mkdir -p %t && cd %t
// RUN: %clang %cflags -g -gsplit-dwarf  \
// RUN:   -fdebug-compilation-dir=/path/does/not/exist %s -o main.exe
// RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -update-debug-sections \
// RUN:   2>&1 | FileCheck %s -check-prefix=DWO-NAME
// RUN: %clang %cflags -g -gsplit-dwarf  \
// RUN:   -fdebug-compilation-dir=/path/does/not/exist %s -o %t/main.exe
// RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -update-debug-sections \
// RUN:   2>&1 | FileCheck %s -check-prefix=DWO-NAME

// DWO-NAME: BOLT-WARNING: Debug Fission: Debug Compilation Dir wrong for

// DWO-NAMET-NOT: Debug Fission: DWO debug information for

