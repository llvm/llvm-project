// UNSUPPORTED: system-windows

// RUN: clang-tidy -checks='-*,clang-analyzer-*' -p %S/Inputs/empty-database %s 2>&1 | FileCheck %s

// CHECK: 'directory' field of compilation database is empty; using the current working directory instead.
