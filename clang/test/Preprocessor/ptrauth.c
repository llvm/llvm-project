// RUN: %clang -E %s --target=aarch64 \
// RUN:   -fptrauth-intrinsics \
// RUN:   -fptrauth-calls \
// RUN:   -fptrauth-returns \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-init-fini | \
// RUN:   FileCheck %s --check-prefixes=INTRIN,CALLS,RETS,VPTR_ADDR_DISCR,VPTR_TYPE_DISCR,INITFINI

// RUN: %clang -E %s --target=aarch64 \
// RUN:   -fptrauth-calls \
// RUN:   -fptrauth-returns \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-init-fini | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,CALLS,RETS,VPTR_ADDR_DISCR,VPTR_TYPE_DISCR,INITFINI

// RUN: %clang -E %s --target=aarch64 \
// RUN:   -fptrauth-intrinsics \
// RUN:   -fptrauth-returns \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-init-fini | \
// RUN:   FileCheck %s --check-prefixes=INTRIN,NOCALLS,RETS,VPTR_ADDR_DISCR,VPTR_TYPE_DISCR,INITFINI

// RUN: %clang -E %s --target=aarch64 \
// RUN:   -fptrauth-intrinsics \
// RUN:   -fptrauth-calls \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-init-fini | \
// RUN:   FileCheck %s --check-prefixes=INTRIN,CALLS,NORETS,VPTR_ADDR_DISCR,VPTR_TYPE_DISCR,INITFINI

// RUN: %clang -E %s --target=aarch64 \
// RUN:   -fptrauth-intrinsics \
// RUN:   -fptrauth-calls \
// RUN:   -fptrauth-returns \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-init-fini | \
// RUN:   FileCheck %s --check-prefixes=INTRIN,CALLS,RETS,NOVPTR_ADDR_DISCR,VPTR_TYPE_DISCR,INITFINI

// RUN: %clang -E %s --target=aarch64 \
// RUN:   -fptrauth-intrinsics \
// RUN:   -fptrauth-calls \
// RUN:   -fptrauth-returns \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-init-fini | \
// RUN:   FileCheck %s --check-prefixes=INTRIN,CALLS,RETS,VPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,INITFINI

// RUN: %clang -E %s --target=aarch64 \
// RUN:   -fptrauth-intrinsics \
// RUN:   -fptrauth-calls \
// RUN:   -fptrauth-returns \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-vtable-pointer-type-discrimination | \
// RUN:   FileCheck %s --check-prefixes=INTRIN,CALLS,RETS,VPTR_ADDR_DISCR,VPTR_TYPE_DISCR,NOINITFINI

#if __has_feature(ptrauth_intrinsics)
// INTRIN: has_ptrauth_intrinsics
void has_ptrauth_intrinsics() {}
#else
// NOINTRIN: no_ptrauth_intrinsics
void no_ptrauth_intrinsics() {}
#endif

#if __has_feature(ptrauth_calls)
// CALLS: has_ptrauth_calls
void has_ptrauth_calls() {}
#else
// NOCALLS: no_ptrauth_calls
void no_ptrauth_calls() {}
#endif

// This is always enabled when ptrauth_calls is enabled
#if __has_feature(ptrauth_member_function_pointer_type_discrimination)
// CALLS: has_ptrauth_member_function_pointer_type_discrimination
void has_ptrauth_member_function_pointer_type_discrimination() {}
#else
// NOCALLS: no_ptrauth_member_function_pointer_type_discrimination
void no_ptrauth_member_function_pointer_type_discrimination() {}
#endif

#if __has_feature(ptrauth_returns)
// RETS: has_ptrauth_returns
void has_ptrauth_returns() {}
#else
// NORETS: no_ptrauth_returns
void no_ptrauth_returns() {}
#endif

#if __has_feature(ptrauth_vtable_pointer_address_discrimination)
// VPTR_ADDR_DISCR: has_ptrauth_vtable_pointer_address_discrimination
void has_ptrauth_vtable_pointer_address_discrimination() {}
#else
// NOVPTR_ADDR_DISCR: no_ptrauth_vtable_pointer_address_discrimination
void no_ptrauth_vtable_pointer_address_discrimination() {}
#endif

#if __has_feature(ptrauth_vtable_pointer_type_discrimination)
// VPTR_TYPE_DISCR: has_ptrauth_vtable_pointer_type_discrimination
void has_ptrauth_vtable_pointer_type_discrimination() {}
#else
// NOVPTR_TYPE_DISCR: no_ptrauth_vtable_pointer_type_discrimination
void no_ptrauth_vtable_pointer_type_discrimination() {}
#endif

#if __has_feature(ptrauth_init_fini)
// INITFINI: has_ptrauth_init_fini
void has_ptrauth_init_fini() {}
#else
// NOINITFINI: no_ptrauth_init_fini
void no_ptrauth_init_fini() {}
#endif
