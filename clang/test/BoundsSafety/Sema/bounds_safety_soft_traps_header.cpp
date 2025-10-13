// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -DMISMATCH -verify=mismatch -verify-ignore-unexpected=note %s
#include <bounds_safety_soft_traps.h>

#ifndef __CLANG_BOUNDS_SAFETY_SOFT_TRAP_API_VERSION
#error macro definition missing
#endif

#if __CLANG_BOUNDS_SAFETY_SOFT_TRAP_API_VERSION > 0
#error API version bumped without updating test
#endif

extern "C" {

// We should get error diagnostics if there's a function signature mismatch
// between the header and the declarations below.
#ifndef MISMATCH
void __bounds_safety_soft_trap_s(const char *reason)
#else
// mismatch-error@+1{{conflicting types for '__bounds_safety_soft_trap_s'}}
void __bounds_safety_soft_trap_s(const char **reason)
#endif
{

}

#ifndef MISMATCH
void __bounds_safety_soft_trap_c(uint16_t reason_code)
#else
// mismatch-error@+1{{conflicting types for '__bounds_safety_soft_trap_c'}}
void __bounds_safety_soft_trap_c(uint32_t reason_code)
#endif
{

}

}

// expected-no-diagnostics
