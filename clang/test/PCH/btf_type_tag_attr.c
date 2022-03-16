// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -include-pch %t \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

int __attribute__((btf_type_tag("tag1"))) __attribute__((btf_type_tag("tag2"))) *p;
