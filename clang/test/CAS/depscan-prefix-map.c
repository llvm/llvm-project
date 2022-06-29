// REQUIRES: clang-cc1daemon
//
// Check with prefix mapping:
//
// RUN: rm -rf %t.d
// RUN: mkdir %t.d
// RUN: %clang -cc1depscan -dump-depscan-tree=%t.root -fdepscan=inline    \
// RUN:    -fdepscan-prefix-map=%S=/^source                               \
// RUN:    -fdepscan-prefix-map=%t.d=/^testdir                            \
// RUN:    -fdepscan-prefix-map=%{objroot}=/^objroot                      \
// RUN:    -fdepscan-prefix-map-toolchain=/^toolchain                     \
// RUN:    -fdepscan-prefix-map-sdk=/^sdk                                 \
// RUN:    -cc1-args -triple x86_64-apple-macos11.0 -x c %s -o %t.d/out.o \
// RUN:              -isysroot %S/Inputs/SDK                              \
// RUN:              -working-directory %t.d                              \
// RUN:              -fcas-path %t.d/cas                                  \
// RUN: | FileCheck %s -DPREFIX=%t.d
// RUN: %clang -cc1depscan -dump-depscan-tree=%t.root -fdepscan=daemon    \
// RUN:    -fdepscan-prefix-map=%S=/^source                               \
// RUN:    -fdepscan-prefix-map=%t.d=/^testdir                            \
// RUN:    -fdepscan-prefix-map=%{objroot}=/^objroot                      \
// RUN:    -fdepscan-prefix-map-toolchain=/^toolchain                     \
// RUN:    -fdepscan-prefix-map-sdk=/^sdk                                 \
// RUN:    -cc1-args -triple x86_64-apple-macos11.0 -x c %s -o %t.d/out.o \
// RUN:              -isysroot %S/Inputs/SDK                              \
// RUN:              -working-directory %t.d                              \
// RUN:              -fcas-path %t.d/cas                                  \
// RUN: | FileCheck %s -DPREFIX=%t.d
//
// CHECK:      "-fcas-path" "[[PREFIX]]/cas"
// CHECK-SAME: "-working-directory" "/^testdir"
// CHECK-SAME: "-x" "c" "/^source/depscan-prefix-map.c"
// CHECK-SAME: "-isysroot" "/^sdk"

// RUN: llvm-cas --cas %t.d/cas --ls-tree-recursive @%t.root              \
// RUN: | FileCheck %s -check-prefix=CHECK-ROOT
//
// RUN: llvm-cas --cas %t.d/cas --ls-tree-recursive @%t.root              \
// RUN: | FileCheck %s -check-prefix=CHECK-ROOT
//
// CHECK-ROOT:      tree
// CHECK-ROOT-SAME:             /^objroot/test/CAS/{{$}}
// CHECK-ROOT-NEXT: tree {{.*}} /^sdk/Library/Frameworks/{{$}}
// CHECK-ROOT-NEXT: file {{.*}} /^source/depscan-prefix-map.c{{$}}

int test() { return 0; }
