// Check target CPUs are correctly passed.

// RUN: %clang -target csky -### -c %s -fsyntax-only 2>&1 -mcpu=ck801 | FileCheck -check-prefix=MCPU-CK801 %s
// MCPU-CK801: "-target-cpu" "ck801"
// MCPU-CK801: "-target-feature" "+elrw" "-target-feature" "+trust" "-target-feature" "+e1"

// RUN: %clang -target csky -### -c %s -fsyntax-only 2>&1 -mcpu=ck801t | FileCheck -check-prefix=MCPU-CK801T %s
// MCPU-CK801T: "-target-cpu" "ck801t"
// MCPU-CK801T: "-target-feature" "+elrw" "-target-feature" "+trust" "-target-feature" "+e1"

// RUN: %clang -target csky -### -c %s -fsyntax-only 2>&1 -mcpu=e801 | FileCheck -check-prefix=MCPU-E801 %s
// MCPU-E801: "-target-cpu" "e801"
// MCPU-E801: "-target-feature" "+elrw" "-target-feature" "+trust" "-target-feature" "+e1"

// RUN: %clang -target csky -### -c %s -fsyntax-only 2>&1 -mcpu=ck802 | FileCheck -check-prefix=MCPU-CK802 %s
// MCPU-CK802: "-target-cpu" "ck802"
// MCPU-CK802: "-target-feature" "+elrw" "-target-feature" "+trust" "-target-feature" "+nvic"
// MCPU-CK802: "-target-feature" "+e1" "-target-feature" "+e2"

// RUN: %clang -target csky -### -c %s -fsyntax-only 2>&1 -mcpu=ck802t | FileCheck -check-prefix=MCPU-CK802T %s
// MCPU-CK802T: "-target-cpu" "ck802t"
// MCPU-CK802T: "-target-feature" "+elrw" "-target-feature" "+trust" "-target-feature" "+nvic"
// MCPU-CK802T: "-target-feature" "+e1" "-target-feature" "+e2"

// TODO: Add more cpu test.
