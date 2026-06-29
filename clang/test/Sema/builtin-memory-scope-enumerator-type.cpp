// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// expected-no-diagnostics

// In C++, enumerators have the enumeration type. Verify using decltype, which
// yields the declared type of the enumerator without any implicit conversion.
static_assert(__is_same(decltype(__memory_scope_system),      __memory_scope), "");
static_assert(__is_same(decltype(__memory_scope_device),      __memory_scope), "");
static_assert(__is_same(decltype(__memory_scope_workgroup),   __memory_scope), "");
static_assert(__is_same(decltype(__memory_scope_wavefront),   __memory_scope), "");
static_assert(__is_same(decltype(__memory_scope_singlethread),__memory_scope), "");
static_assert(__is_same(decltype(__memory_scope_cluster),     __memory_scope), "");
