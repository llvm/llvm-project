// Verify that the modules driver rejects ambiguous module definitions.

// RUN: split-file %s %t

// RUN: not %clang -std=c++23 -fmodules -fmodules-driver -Rmodules-driver \
// RUN:   %t/main.cpp %t/A1.cpp %t/A2.cpp 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t --check-prefixes=CHECK %s

/// CHECK: clang: error: duplicate definitions of C++20 named module 'A' in '[[PREFIX]]/A1.cpp' and '[[PREFIX]]/A2.cpp'

//--- main.cpp
import A;

//--- A1.cpp
export module A;

//--- A2.cpp
export module A;
