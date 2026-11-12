// RUN: not %clang_cc1 -std=c++20 -fcxx-profiles=std::casting -fsyntax-only \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=NOT-ENABLED
// RUN: not %clang_cc1 -std=c++20 -fcxx-profiles-preview=std::casting \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=PREVIEW-NOT-ENABLED
// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles \
// RUN:   -fcxx-profiles=acme::thing -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNKNOWN
// RUN: not %clang_cc1 -std=c++20 -fexperimental-cxx-profiles \
// RUN:   -fcxx-profiles=std::casting -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISMATCH

// NOT-ENABLED: error: '-fcxx-profiles=' requires '-fexperimental-cxx-profiles'
// PREVIEW-NOT-ENABLED: error: '-fcxx-profiles-preview=' requires '-fexperimental-cxx-profiles'
// UNKNOWN: warning: unknown profile 'acme::thing'; it will not be enforced

// MISMATCH: error: profile 'std::casting' is already enforced with designator 'std::casting'
// MISMATCH: note: profile 'std::casting' was enforced on the command line
[[profiles::enforce(std::casting(strict))]];
