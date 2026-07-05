// Test that system-headerness works for building modules.

// RUN: rm -rf %t
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -isystem %S/Inputs -pedantic -Werror %s -verify -std=c11
// expected-no-diagnostics

@import warning;
int i = bigger_than_int;

#include <stddef.h>

#define __need_size_t
#include <stddef.h>
