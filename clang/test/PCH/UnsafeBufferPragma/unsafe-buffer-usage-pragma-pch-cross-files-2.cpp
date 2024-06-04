
// Test with PCH: See `header-2.hpp` for what are being tested.
//
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t.pch %S/header-2.hpp
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t.pch -verify %s
// RUN: rm %t.pch

// expected-warning@header-2.hpp:5 {{unsafe buffer access}}
// expected-note@header-2.hpp:5 {{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
