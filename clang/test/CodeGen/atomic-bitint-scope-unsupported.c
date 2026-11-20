// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=256 -Wno-atomic-alignment -emit-llvm -verify -DUPDATE %s -o /dev/null
// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=256 -Wno-atomic-alignment -emit-llvm -verify -DDYNAMIC_UPDATE %s -o /dev/null
// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=256 -Wno-atomic-alignment -emit-llvm -verify -DLOAD %s -o /dev/null
// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=256 -Wno-atomic-alignment -emit-llvm -verify -DSTORE %s -o /dev/null
// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=256 -Wno-atomic-alignment -emit-llvm -verify -DEXCHANGE %s -o /dev/null
// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=256 -Wno-atomic-alignment -emit-llvm -verify -DCMPXCHG %s -o /dev/null
// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=256 -Wno-atomic-alignment -emit-llvm -verify -DDYNAMIC_LOAD %s -o /dev/null
// RUN: %clang_cc1 -std=c23 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=256 -Wno-atomic-alignment -emit-llvm -DSYSTEM_LOAD %s -o - | FileCheck --check-prefix=SYSTEM %s

typedef unsigned _BitInt(65) U65;
typedef unsigned _BitInt(129) U129;

#if !defined(SYSTEM_LOAD)
// expected-error@*:* {{scoped _BitInt atomic operation is not supported at this width}}
#endif
#if defined(UPDATE)
U65 scoped_add65(U65 *p) {
  return __scoped_atomic_fetch_add(p, (U65)1, __ATOMIC_RELAXED,
                                   __MEMORY_SCOPE_WRKGRP);
}
#elif defined(DYNAMIC_UPDATE)
U65 scoped_add65_dynamic(U65 *p, int scope) {
  return __scoped_atomic_fetch_add(p, (U65)1, __ATOMIC_RELAXED, scope);
}
#elif defined(LOAD)
void scoped_load129(U129 *p, U129 *out) {
  __scoped_atomic_load(p, out, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
}
#elif defined(STORE)
void scoped_store129(U129 *p, U129 *value) {
  __scoped_atomic_store(p, value, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
}
#elif defined(EXCHANGE)
void scoped_exchange129(U129 *p, U129 *value, U129 *out) {
  __scoped_atomic_exchange(p, value, out, __ATOMIC_RELAXED,
                           __MEMORY_SCOPE_WRKGRP);
}
#elif defined(CMPXCHG)
_Bool scoped_cmpxchg129(U129 *p, U129 *expected, U129 *desired) {
  return __scoped_atomic_compare_exchange(p, expected, desired, 0,
                                          __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                          __MEMORY_SCOPE_WRKGRP);
}
#elif defined(DYNAMIC_LOAD)
void scoped_load129_dynamic(U129 *p, U129 *out, int scope) {
  __scoped_atomic_load(p, out, __ATOMIC_RELAXED, scope);
}
#elif defined(SYSTEM_LOAD)
// SYSTEM-LABEL: define {{.*}} @scoped_load129_system(
// SYSTEM: call void @__atomic_load(i64 noundef 24,
void scoped_load129_system(U129 *p, U129 *out) {
  __scoped_atomic_load(p, out, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}
#endif
