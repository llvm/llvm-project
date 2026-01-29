// RUN: %clang_cc1 -E -P --show-includes -verify=suppressed \
// RUN:   -fcase-insensitive-paths -Wno-nonportable-include-path %s 2>&1 \
// RUN:   | FileCheck %s
// Make sure the exact spelling used in the #include directive is preserved
// when printing header dependencies (consistent with MSVC /showIncludes).
// CHECK: Note: including file: {{.*}}inPuts/eMptY.H
// suppressed-no-diagnostics@+1
#include "inPuts/eMptY.H"
// expected-warning@-1 {{non-portable path to file '"Inputs/empty.h"'; specified path differs in case from file name on disk}}
// RUN: %clang_cc1 -E -P -verify -fcase-insensitive-paths %s
