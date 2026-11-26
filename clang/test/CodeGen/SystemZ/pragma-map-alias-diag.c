// RUN: %clang_cc1 %s -triple s390x-ibm-zos -emit-llvm-only -verify
// REQUIRES: systemz-registered-target

// Once the aliasee is mapped, its symbol name is the mapped name, so an alias
// that names it by its source name no longer refers to a defined symbol.
#pragma map(t1, "T1")
void t1(void) {}
void a1(void) __attribute__((alias("t1"))); // expected-error {{alias must point to a defined variable or function}} \
                                            // expected-note {{the function or variable specified in an alias must refer to its mangled name}} \
                                            // expected-note {{function by that name is mangled as "T1"}}

#pragma map(v2, "V2")
int v2 = 2;
extern int va2 __attribute__((alias("v2"))); // expected-error {{alias must point to a defined variable or function}} \
                                             // expected-note {{the function or variable specified in an alias must refer to its mangled name}} \
                                             // expected-note {{function by that name is mangled as "V2"}}
