// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// expected-no-diagnostics

static_assert(!__is_pod(void), "");
static_assert(!__is_pod(int&), "");
static_assert(!__is_pod(int()), "");

static_assert(!__is_trivially_copyable(void), "");
static_assert(!__is_trivially_copyable(int&), "");
static_assert(!__is_trivially_copyable(int()), "");

static_assert(!__is_trivially_relocatable(void), "");
static_assert(!__is_trivially_relocatable(int&), "");
static_assert(!__is_trivially_relocatable(int()), "");
