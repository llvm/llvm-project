// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=nullability.NullReturnedFromNonnull

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=osx.cocoa.RetainCount \
// RUN:   -analyzer-list-enabled-checkers \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-IMPLICITLY-ENABLED

// CHECK-IMPLICITLY-ENABLED: osx.cocoa.RetainCountBase
// CHECK-IMPLICITLY-ENABLED: osx.cocoa.RetainCount

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=osx.cocoa.RetainCount \
// RUN:   -analyzer-disable-checker=osx.cocoa.RetainCountBase \
// RUN:   -analyzer-list-enabled-checkers \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-IMPLICITLY-DISABLED

// CHECK-IMPLICITLY-DISABLED-NOT: osx.cocoa.RetainCountBase
// CHECK-IMPLICITLY-DISABLED-NOT: osx.cocoa.RetainCount
