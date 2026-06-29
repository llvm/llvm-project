// RUN: %clang_cc1 -fsyntax-only -std=c11 -verify %s
// expected-no-diagnostics

// In C, enumerators have type int (C11 6.7.2.2p3), not the enumeration type.
// Verify using _Generic, which selects based on the type of the controlling
// expression without any implicit conversion.
_Static_assert(_Generic(__memory_scope_system,      int: 1, default: 0), "");
_Static_assert(_Generic(__memory_scope_device,      int: 1, default: 0), "");
_Static_assert(_Generic(__memory_scope_workgroup,   int: 1, default: 0), "");
_Static_assert(_Generic(__memory_scope_wavefront,   int: 1, default: 0), "");
_Static_assert(_Generic(__memory_scope_singlethread,int: 1, default: 0), "");
_Static_assert(_Generic(__memory_scope_cluster,     int: 1, default: 0), "");
