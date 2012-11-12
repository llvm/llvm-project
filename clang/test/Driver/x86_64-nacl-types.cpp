// RUN: %clang_cc1 -triple x86_64-unknown-nacl -std=c++11 -verify %s
// expected-no-diagnostics

#include <stddef.h>
#include <stdarg.h>

static_assert(alignof(char) == 1, "alignof char is wrong");

static_assert(alignof(short) == 2, "sizeof short is wrong");
static_assert(alignof(short) == 2, "alignof short is wrong");

static_assert(alignof(int) == 4, "sizeof int is wrong");
static_assert(alignof(int) == 4, "alignof int is wrong");

static_assert(sizeof(long) == 4, "sizeof long is wrong");
static_assert(sizeof(long) == 4, "alignof long is wrong");

static_assert(sizeof(long long) == 8, "sizeof long long is wrong wrong");
static_assert(alignof(long long) == 8, "alignof long long is wrong wrong");

static_assert(sizeof(void*) == 4, "sizeof void * is wrong");
static_assert(alignof(void*) == 4, "alignof void * is wrong");

static_assert(sizeof(float) == 4, "sizeof float is wrong");
static_assert(alignof(float) == 4, "alignof float is wrong");

static_assert(sizeof(double) == 8, "sizeof double is wrong");
static_assert(alignof(double) == 8, "alignof double is wrong");

static_assert(sizeof(long double) == 8, "sizeof long double is wrong");
static_assert(alignof(long double) == 8, "alignof long double is wrong");

static_assert(sizeof(va_list) == 16, "sizeof va_list is wrong");
static_assert(alignof(va_list) == 4, "alignof va_list is wrong");

static_assert(sizeof(size_t) == 4, "sizeof size_t is wrong");
static_assert(alignof(size_t) == 4, "alignof size_t is wrong");
