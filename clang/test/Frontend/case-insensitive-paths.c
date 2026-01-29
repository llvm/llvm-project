// RUN: %clang_cc1 -E -P -verify --show-includes -Wno-nonportable-include-path \
// RUN:     -fcase-insensitive-paths %s 2>&1 | FileCheck %s

#include "inPuts/eMptY.H" // expected-no-diagnostics

// Make sure the exact spelling used in the #include directive is preserved
// when printing header dependencies (consistent with MSVC /showIncludes).
// CHECK: Note: including file: {{.*}}inPuts/eMptY.H
