// RUN: %clang_cc1 -analyzer-checker-help \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-STABLE

// RUN: %clang_cc1 -analyzer-checker-help-alpha \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-ALPHA

// RUN: %clang_cc1 -analyzer-checker-help-developer \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-DEVELOPER

// RUN: %clang_cc1 -analyzer-checker-help-developer \
// RUN:   -analyzer-checker-help-alpha \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-DEVELOPER-ALPHA

// RUN: %clang_cc1 -analyzer-checker-help \
// RUN:   -analyzer-checker-help-alpha \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-STABLE-ALPHA

// RUN: %clang_cc1 -analyzer-checker-help \
// RUN:   -analyzer-checker-help-developer \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-STABLE-DEVELOPER

// RUN: %clang_cc1 -analyzer-checker-help \
// RUN:   -analyzer-checker-help-alpha \
// RUN:   -analyzer-checker-help-developer \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-STABLE-ALPHA-DEVELOPER

// Note that alpha.cplusplus.IteratorModeling is not only an alpha, but also a
// hidden checker. In this case, we'd only like to see it in the developer list.
// CHECK-ALPHA-NOT: alpha.cplusplus.IteratorModeling
// CHECK-DEVELOPER: alpha.cplusplus.IteratorModeling

// CHECK-STABLE:        core.DivideZero
// CHECK-DEVELOPER-NOT: core.DivideZero
// CHECK-ALPHA-NOT:     core.DivideZero

// CHECK-STABLE-NOT: debug.ConfigDumper
// CHECK-DEVELOPER:  debug.ConfigDumper
// CHECK-ALPHA-NOT:  debug.ConfigDumper


// CHECK-STABLE-ALPHA:        core.DivideZero
// CHECK-DEVELOPER-ALPHA-NOT: core.DivideZero
// CHECK-STABLE-DEVELOPER:    core.DivideZero

// CHECK-STABLE-ALPHA-NOT: debug.ConfigDumper
// CHECK-DEVELOPER-ALPHA:  debug.ConfigDumper
// CHECK-STABLE-DEVELOPER: debug.ConfigDumper


// CHECK-STABLE-ALPHA-DEVELOPER: core.DivideZero
// CHECK-STABLE-ALPHA-DEVELOPER: debug.ConfigDumper
