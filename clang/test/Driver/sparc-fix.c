// RUN: %clang --target=sparc -mfix-gr712rc -### %s 2>&1 | FileCheck --check-prefix=GR712RC %s
// GR712RC: "-target-feature" "+fix-tn0009" "-target-feature" "+fix-tn0011" "-target-feature" "+fix-tn0012" "-target-feature" "+fix-tn0013"

// RUN: %clang --target=sparc -mfix-ut700 -### %s 2>&1 | FileCheck --check-prefix=UT700 %s
// UT700: "-target-feature" "+fix-tn0009" "-target-feature" "+fix-tn0010" "-target-feature" "+fix-tn0013"
