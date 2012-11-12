// RUN: %clang_cc1 -ferror-limit 1 -fsyntax-only %s 2>&1 | FileCheck %s

// error and note emitted
struct s1{};
struct s1{};

// error and note suppressed by error-limit
struct s2{};
struct s2{};

// CHECK: 5:8: error: redefinition of 's1'
// CHECK: 4:8: note: previous definition is here
// CHECK: fatal error: too many errors emitted, stopping now
// CHECK-NOT: 9:8: error: redefinition of 's2'
// CHECK-NOT: 8:8: note: previous definition is here
