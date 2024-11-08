//// Note: preprocessor features exactly match corresponding clang driver flags. However, some flags are only intended to be used in combination with other ones.
//// For example, -fptrauth-init-fini will not affect codegen without -fptrauth-calls, but the preprocessor feature would be set anyway.

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-intrinsics | \
// RUN:   FileCheck %s --check-prefixes=INTRIN,NOCALLS,NORETS,NOVPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,NOTYPE_INFO_DISCR,NOFUNC,NOINITFINI,NOINITFINI_ADDR_DISCR,NOGOTOS

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-calls | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,CALLS,NORETS,NOVPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,NOTYPE_INFO_DISCR,NOFUNC,NOINITFINI,NOINITFINI_ADDR_DISCR,NOGOTOS

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-returns | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,NOCALLS,RETS,NOVPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,NOTYPE_INFO_DISCR,NOFUNC,NOINITFINI,NOINITFINI_ADDR_DISCR,NOGOTOS

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-vtable-pointer-address-discrimination | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,NOCALLS,NORETS,VPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,NOTYPE_INFO_DISCR,NOFUNC,NOINITFINI,NOINITFINI_ADDR_DISCR,NOGOTOS

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-vtable-pointer-type-discrimination | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,NOCALLS,NORETS,NOVPTR_ADDR_DISCR,VPTR_TYPE_DISCR,NOTYPE_INFO_DISCR,NOFUNC,NOINITFINI,NOINITFINI_ADDR_DISCR,NOGOTOS

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-type-info-vtable-pointer-discrimination | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,NOCALLS,NORETS,NOVPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,TYPE_INFO_DISCR,NOFUNC,NOINITFINI,NOINITFINI_ADDR_DISCR,NOGOTOS

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-function-pointer-type-discrimination | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,NOCALLS,NORETS,NOVPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,NOTYPE_INFO_DISCR,FUNC,NOINITFINI,NOINITFINI_ADDR_DISCR,NOGOTOS

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-init-fini | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,NOCALLS,NORETS,NOVPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,NOTYPE_INFO_DISCR,NOFUNC,INITFINI,NOINITFINI_ADDR_DISCR,NOGOTOS

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-init-fini-address-discrimination | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,NOCALLS,NORETS,NOVPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,NOTYPE_INFO_DISCR,NOFUNC,NOINITFINI,INITFINI_ADDR_DISCR,NOGOTOS

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-indirect-gotos | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN,NOCALLS,NORETS,NOVPTR_ADDR_DISCR,NOVPTR_TYPE_DISCR,NOTYPE_INFO_DISCR,NOFUNC,NOINITFINI,NOINITFINI_ADDR_DISCR,GOTOS

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

#if __has_feature(ptrauth_type_info_vtable_pointer_discrimination)
// TYPE_INFO_DISCR: has_ptrauth_type_info_vtable_pointer_discrimination
void has_ptrauth_type_info_vtable_pointer_discrimination() {}
#else
// NOTYPE_INFO_DISCR: no_ptrauth_type_info_vtable_pointer_discrimination
void no_ptrauth_type_info_vtable_pointer_discrimination() {}
#endif

#if __has_feature(ptrauth_function_pointer_type_discrimination)
// FUNC: has_ptrauth_function_pointer_type_discrimination
void has_ptrauth_function_pointer_type_discrimination() {}
#else
// NOFUNC: no_ptrauth_function_pointer_type_discrimination
void no_ptrauth_function_pointer_type_discrimination() {}
#endif

#if __has_feature(ptrauth_init_fini)
// INITFINI: has_ptrauth_init_fini
void has_ptrauth_init_fini() {}
#else
// NOINITFINI: no_ptrauth_init_fini
void no_ptrauth_init_fini() {}
#endif

#if __has_feature(ptrauth_init_fini_address_discrimination)
// INITFINI_ADDR_DISCR: has_ptrauth_init_fini_address_discrimination
void has_ptrauth_init_fini_address_discrimination() {}
#else
// NOINITFINI_ADDR_DISCR: no_ptrauth_init_fini_address_discrimination
void no_ptrauth_init_fini_address_discrimination() {}
#endif

#if __has_feature(ptrauth_indirect_gotos)
// GOTOS: has_ptrauth_indirect_gotos
void has_ptrauth_indirect_gotos() {}
#else
// NOGOTOS: no_ptrauth_indirect_gotos
void no_ptrauth_indirect_gotos() {}
#endif
