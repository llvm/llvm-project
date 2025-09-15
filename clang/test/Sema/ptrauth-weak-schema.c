// RUN: %clang_cc1 -triple arm64e-apple-ios -fptrauth-calls -fptrauth-intrinsics -fsyntax-only -Wno-unused-variable -verify %s
// RUN: %clang_cc1 -triple arm64e-apple-ios -DNO_PTRAUTH -fsyntax-only -Wno-unused-variable -verify=noptrauth %s

// noptrauth-no-diagnostics

#include <ptrauth.h>

#if defined(__PTRAUTH__) == defined(NO_PTRAUTH)
#error expected pointer authentication state does not match actual
#endif

#if defined(NO_PTRAUTH)
#define FN_PTR_AUTH(address_diversity, constant_discriminator)
#else
#define FN_PTR_AUTH(address_diversity, constant_discriminator) \
  __ptrauth(ptrauth_key_function_pointer, address_diversity, constant_discriminator)
#endif

// Global variables with external linkage and weak pointer authentication should
// not raise any warning.
extern void(* g1_external_weak)(void);
void(* FN_PTR_AUTH(0, 0) g2_external_weak)(void);

// Global variables with internal linkage and strong pointer authentication
// should not raise any warning.
static void(* FN_PTR_AUTH(1, 65535) g1_internal_strong)(void);
static void(* FN_PTR_AUTH(0, 65535) g2_internal_strong)(void);
static void(* FN_PTR_AUTH(1, 0) g3_internal_strong)(void);

#if !defined(NO_PTRAUTH)
// Global variables with internal linkage and weak pointer authentication should
// raise a warning.
static void(* g1_internal_weak)(void);
// expected-warning@-1 {{'g1_internal_weak' has internal linkage with a default pointer authentication schema that should be overridden by an explicit schema with unique diversifiers}}
static void(* FN_PTR_AUTH(0, 0) g2_internal_weak)(void);
// expected-warning@-1 {{'g2_internal_weak' has internal linkage with a pointer authentication schema that should be overridden by a schema with unique diversifiers}}

// Assert that -Wptrauth-weak-schema silences warnings.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wptrauth-weak-schema"
static void(* g3_internal_weak)(void);
#pragma clang diagnostic pop
#endif

void test_local_variables(void) {
    #if !defined(NO_PTRAUTH)
    // Local variables (internal linkage) with weak pointer authentication
    // should raise a warning.
    static void(* l1_internal_weak)(void);
    // expected-warning@-1 {{'l1_internal_weak' has internal linkage with a default pointer authentication schema that should be overridden by an explicit schema with unique diversifiers}}
    static void(* FN_PTR_AUTH(0, 0) l2_internal_weak)(void);
    // expected-warning@-1 {{'l2_internal_weak' has internal linkage with a pointer authentication schema that should be overridden by a schema with unique diversifiers}}
    void(* l3_internal_weak)(void);
    // expected-warning@-1 {{'l3_internal_weak' has internal linkage with a default pointer authentication schema that should be overridden by an explicit schema with unique diversifiers}}
    void(* FN_PTR_AUTH(0, 0) l4_internal_weak)(void);
    // expected-warning@-1 {{'l4_internal_weak' has internal linkage with a pointer authentication schema that should be overridden by a schema with unique diversifiers}}
    #endif

    // Local variables (internal linkage) with strong pointer authentication
    // should not raise any warning.
    static void(* FN_PTR_AUTH(1, 65535) l1_internal_strong)(void);
    static void(* FN_PTR_AUTH(0, 65535) l2_internal_strong)(void);
    static void(* FN_PTR_AUTH(1, 0) l3_internal_strong)(void);
    void(* FN_PTR_AUTH(1, 65535) l4_internal_strong)(void);
    void(* FN_PTR_AUTH(0, 65535) l5_internal_strong)(void);
    void(* FN_PTR_AUTH(1, 0) l6_internal_strong)(void);
}
