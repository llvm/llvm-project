// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// Show that we suppress the reserved identifier diagnostic in a system header.
# 100 "file.cpp" 1 3  // Enter a system header
export module __test;
# 100 "file.cpp" 2 3  // Leave the system header
