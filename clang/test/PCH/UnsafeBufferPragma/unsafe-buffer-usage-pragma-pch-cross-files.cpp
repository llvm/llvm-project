// Test with PCH: See `header.hpp` for what are being tested.
//
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t.pch %S/header.hpp
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t.pch -verify %s
// RUN: rm %t.pch

// expected-warning@header.hpp:6 {{unsafe buffer access}}
// expected-note@header.hpp:6 {{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
