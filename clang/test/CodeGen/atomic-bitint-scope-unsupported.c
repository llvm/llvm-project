// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -Wno-atomic-alignment -emit-llvm -verify %s -o /dev/null
// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -Wno-atomic-alignment -emit-llvm -verify -DDYNAMIC %s -o /dev/null

typedef unsigned _BitInt(65) U65;

// expected-error@*:* {{scoped _BitInt atomic operation is not supported at this width}}
#ifndef DYNAMIC
U65 scoped_add65(U65 *p) {
  return __scoped_atomic_fetch_add(p, (U65)1, __ATOMIC_RELAXED,
                                   __MEMORY_SCOPE_WRKGRP);
}
#else
U65 scoped_add65_dynamic(U65 *p, int scope) {
  return __scoped_atomic_fetch_add(p, (U65)1, __ATOMIC_RELAXED, scope);
}
#endif
