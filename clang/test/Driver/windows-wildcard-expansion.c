// Clang does wildcard expansion on Windows. On other OSs, it's done by the shell.
// REQUIRES: system-windows

// RUN: %clang -c -### %S/Inputs/wildcard*.c 2>&1 | FileCheck %s
// RUN: %clang -c -### %S/Inputs/wildcard?.c 2>&1 | FileCheck %s
// CHECK: wildcard1.c
// CHECK: wildcard2.c
