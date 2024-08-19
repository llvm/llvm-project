// RUN: %clang -target sparc -mfix-gr712rc -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIX-GR712RC < %t %s
// CHECK-FIX-GR712RC: "-target-feature" "+fix-tn0009"
// CHECK-FIX-GR712RC: "-target-feature" "+fix-tn0011"
// CHECK-FIX-GR712RC: "-target-feature" "+fix-tn0012"
// CHECK-FIX-GR712RC: "-target-feature" "+fix-tn0013"

// RUN: %clang -target sparc -mfix-ut700 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIX-UT700 < %t %s
// CHECK-FIX-UT700: "-target-feature" "+fix-tn0009"
// CHECK-FIX-UT700: "-target-feature" "+fix-tn0010"
// CHECK-FIX-UT700: "-target-feature" "+fix-tn0013"
