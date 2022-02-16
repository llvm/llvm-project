// RUN: %clang_cc1                -debug-info-kind=limited -emit-llvm %p/Inputs/debug-info-embed-source.c -o - | FileCheck %s --check-prefix=NOEMBED
// RUN: %clang_cc1 -gembed-source -debug-info-kind=limited -emit-llvm %p/Inputs/debug-info-embed-source.c -o - | FileCheck %s --check-prefix=EMBED

// NOEMBED-NOT: !DIFile({{.*}}source:
// EMBED: !DIFile({{.*}}source: "void foo(void) { }\0A"
