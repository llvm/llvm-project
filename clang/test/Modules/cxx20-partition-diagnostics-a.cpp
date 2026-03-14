// Module Partition diagnostics

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/bad-import.cpp -verify

// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/bad-partition.cpp -verify

//--- bad-import.cpp

import :B; // expected-error {{module partition imports must be within a module purview}}

//--- bad-partition.cpp

module; // expected-error {{missing 'module' declaration at end of global module fragment introduced here}}

import :Part; // expected-error {{module partition imports cannot be in the global module fragment}}
