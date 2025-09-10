// RUN: %clang_cc1 -triple arm64e-apple-ios -fptrauth-calls -fptrauth-intrinsics -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64e-apple-ios -DNO_PTRAUTH -fsyntax-only -verify %s

#if defined(NO_PTRAUTH)

#define FN_PTR_AUTH(address_diversity, constant_discriminator)
// expected-no-diagnostics

#else // !defined(NO_PTRAUTH)

#if !__has_extension(ptrauth_qualifier)
#error __ptrauth qualifier not enabled
#endif

#include <ptrauth.h>

#define FN_PTR_AUTH(address_diversity, constant_discriminator) \
  __ptrauth(ptrauth_key_function_pointer, address_diversity, constant_discriminator)

#endif // defined(NO_PTRAUTH)

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
// expected-warning@-1 {{internal variable 'g1_internal_weak' is using a weak signing schema for pointer authentication}}
static void(* FN_PTR_AUTH(0, 0) g2_internal_weak)(void);
// expected-warning@-1 {{internal variable 'g2_internal_weak' is using a weak signing schema for pointer authentication}}

// Assert that -Wptrauth-weak-schema silences warnings.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wptrauth-weak-schema"
static void(* g3_internal_weak)(void);
#pragma clang diagnostic pop
#endif

void test_local_variables(void) {
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-variable"

    #if !defined(NO_PTRAUTH)
    // Local variables (internal linkage) with weak pointer authentication
    // should raise a warning.
    static void(* l1_internal_weak)(void);
    // expected-warning@-1 {{internal variable 'l1_internal_weak' is using a weak signing schema for pointer authentication}}
    static void(* FN_PTR_AUTH(0, 0) l2_internal_weak)(void);
    // expected-warning@-1 {{internal variable 'l2_internal_weak' is using a weak signing schema for pointer authentication}}
    #endif

    // Local variables (internal linkage) with strong pointer authentication
    // should not raise any warning.
    void(* FN_PTR_AUTH(1, 65535) l1_internal_strong)(void);
    void(* FN_PTR_AUTH(0, 65535) l2_internal_strong)(void);
    void(* FN_PTR_AUTH(1, 0) l3_internal_strong)(void);

    #pragma clang diagnostic pop
}
