// RUN: rm -rf %t && mkdir -p %t

// RUN: not %clang_cc1 -triple arm64-apple-macosx12 -fsyntax-only %s 2> %t/diags-orig

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/t.rsp -cc1-args \
// RUN:   -cc1 -triple arm64-apple-macosx12 -fcas-path %t/cas -fsyntax-only %s
// RUN: not %clang @%t/t.rsp 2> %t/diags-cached

// RUN: diff -u %t/diags-orig %t/diags-cached

// RUN: FileCheck %s -input-file %t/diags-cached

const char s8[] = @encode(__SVInt8_t);
// CHECK: cannot yet @encode type __SVInt8_t
