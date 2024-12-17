// RUN: %clang -Wunused-command-line-argument -frealloc-lhs -### %s 2> %t
// RUN: FileCheck < %t %s --check-prefix=REALLOCLHS
// RUN: %clang -Wunused-command-line-argument -fno-realloc-lhs -### %s 2> %t
// RUN: FileCheck < %t %s --check-prefix=NOREALLOCLHS

// CHECK: argument unused during compilation: '-frealloc-lhs'
// CHECK: argument unused during compilation: '-fno-realloc-lhs'
