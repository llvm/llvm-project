// RUN: %clang -target csky-unknown-elf -march=ck801 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK801

// CHECK-CK801: "-target-cpu" "ck801"
// CHECK-CK801: "-target-feature" "+elrw" "-target-feature" "+trust"
// CHECK-CK801: "-target-feature" "+e1"

// RUN: %clang -target csky-unknown-elf -march=ck802 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK802

// CHECK-CK802: "-target-cpu" "ck802"
// CHECK-CK802: "-target-feature" "+elrw" "-target-feature" "+trust"
// CHECK-CK802: "-target-feature" "+nvic" "-target-feature" "+e1"
// CHECK-CK802: "-target-feature" "+e2"

// RUN: %clang -target csky-unknown-elf -march=ck803 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK803

// CHECK-CK803: "-target-cpu" "ck803"
// CHECK-CK803: "-target-feature" "+hwdiv" "-target-feature" "+elrw"
// CHECK-CK803: "-target-feature" "+trust" "-target-feature" "+nvic"
// CHECK-CK803: "-target-feature" "+e1" "-target-feature" "+e2" "-target-feature" "+2e3"
// CHECK-CK803: "-target-feature" "+mp"

// RUN: %clang -target csky-unknown-elf -march=ck803s -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK803S

// CHECK-CK803S: "-target-cpu" "ck803s"
// CHECK-CK803S: "-target-feature" "+hwdiv" "-target-feature" "+elrw"
// CHECK-CK803S: "-target-feature" "+trust" "-target-feature" "+nvic"
// CHECK-CK803S: "-target-feature" "+e1" "-target-feature" "+e2"
// CHECK-CK803S: "-target-feature" "+2e3" "-target-feature" "+mp"

// RUN: %clang -target csky-unknown-elf -march=ck804 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK804

// CHECK-CK804: "-target-cpu" "ck804"
// CHECK-CK804: "-target-feature" "+hwdiv" "-target-feature" "+elrw"
// CHECK-CK804: "-target-feature" "+trust" "-target-feature" "+nvic"
// CHECK-CK804: "-target-feature" "+doloop" "-target-feature" "+e1"
// CHECK-CK804: "-target-feature" "+e2" "-target-feature" "+2e3"
// CHECK-CK804: "-target-feature" "+mp" "-target-feature" "+3e3r1"
// CHECK-CK804: "-target-feature" "+3e3r2" "-target-feature" "+3e3r3"

// RUN: %clang -target csky-unknown-elf -march=ck805 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK805

// CHECK-CK805: "-target-cpu" "ck805"
// CHECK-CK805: "-target-feature" "+hwdiv" "-target-feature" "+elrw"
// CHECK-CK805: "-target-feature" "+trust" "-target-feature" "+nvic"
// CHECK-CK805: "-target-feature" "+doloop" "-target-feature" "+high-registers"
// CHECK-CK805: "-target-feature" "+vdsp2e3" "-target-feature" "+vdspv2" "-target-feature" "+e1"
// CHECK-CK805: "-target-feature" "+e2" "-target-feature" "+2e3" "-target-feature" "+mp"
// CHECK-CK805: "-target-feature" "+3e3r1" "-target-feature" "+3e3r2" "-target-feature" "+3e3r3"

// RUN: %clang -target csky-unknown-elf -march=ck807 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK807

// CHECK-CK807: "-target-cpu" "ck807"
// CHECK-CK807: "-target-feature" "+hwdiv" "-target-feature" "+edsp"
// CHECK-CK807: "-target-feature" "+dsp1e2" "-target-feature" "+dspe60" "-target-feature" "+elrw"
// CHECK-CK807: "-target-feature" "+trust" "-target-feature" "+cache" "-target-feature" "+nvic"
// CHECK-CK807: "-target-feature" "+high-registers" "-target-feature" "+hard-tp" "-target-feature" "+e1"
// CHECK-CK807: "-target-feature" "+e2" "-target-feature" "+2e3" "-target-feature" "+mp"
// CHECK-CK807: "-target-feature" "+3e7" "-target-feature" "+mp1e2"

// RUN: %clang -target csky-unknown-elf -march=ck810 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK810

// CHECK-CK810: "-target-cpu" "ck810"
// CHECK-CK810: "-target-feature" "+hwdiv" "-target-feature" "+edsp" "-target-feature" "+dsp1e2"
// CHECK-CK810: "-target-feature" "+dspe60" "-target-feature" "+elrw" "-target-feature" "+trust"
// CHECK-CK810: "-target-feature" "+cache" "-target-feature" "+nvic" "-target-feature" "+high-registers"
// CHECK-CK810: "-target-feature" "+hard-tp" "-target-feature" "+e1" "-target-feature" "+e2" "-target-feature" "+2e3"
// CHECK-CK810: "-target-feature" "+mp" "-target-feature" "+3e7" "-target-feature" "+mp1e2" "-target-feature" "+7e10"

// RUN: %clang -target csky-unknown-elf -march=ck810v -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK810V

// CHECK-CK810V: "-target-cpu" "ck810v"
// CHECK-CK810V: "-target-feature" "+hwdiv" "-target-feature" "+edsp" "-target-feature" "+dsp1e2"
// CHECK-CK810V: "-target-feature" "+dspe60" "-target-feature" "+elrw" "-target-feature" "+trust"
// CHECK-CK810V: "-target-feature" "+cache" "-target-feature" "+nvic" "-target-feature" "+high-registers"
// CHECK-CK810V: "-target-feature" "+hard-tp" "-target-feature" "+vdspv1" "-target-feature" "+e1"
// CHECK-CK810V: "-target-feature" "+e2" "-target-feature" "+2e3" "-target-feature" "+mp"
// CHECK-CK810V: "-target-feature" "+3e7" "-target-feature" "+mp1e2" "-target-feature" "+7e10"

// RUN: %clang -target csky-unknown-elf -march=ck860 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK860

// CHECK-CK860: "-target-cpu" "ck860"
// CHECK-CK860: "-target-feature" "+hwdiv" "-target-feature" "+dspe60" "-target-feature" "+elrw"
// CHECK-CK860: "-target-feature" "+trust" "-target-feature" "+cache" "-target-feature" "+nvic"
// CHECK-CK860: "-target-feature" "+doloop" "-target-feature" "+high-registers" "-target-feature" "+hard-tp"
// CHECK-CK860: "-target-feature" "+e1" "-target-feature" "+e2" "-target-feature" "+2e3" "-target-feature" "+mp"
// CHECK-CK860: "-target-feature" "+3e3r1" "-target-feature" "+3e3r2" "-target-feature" "+3e3r3"
// CHECK-CK860: "-target-feature" "+3e7" "-target-feature" "+mp1e2" "-target-feature" "+7e10" "-target-feature" "+10e60"

// RUN: %clang -target csky-unknown-elf -march=ck860v -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s -check-prefixes=CHECK-CK860V

// CHECK-CK860V: "-target-cpu" "ck860v"
// CHECK-CK860V: "-target-feature" "+hwdiv" "-target-feature" "+dspe60" "-target-feature" "+elrw" "-target-feature" "+trust"
// CHECK-CK860V: "-target-feature" "+cache" "-target-feature" "+nvic" "-target-feature" "+doloop"
// CHECK-CK860V: "-target-feature" "+high-registers" "-target-feature" "+vdsp2e60f" "-target-feature" "+vdspv2"
// CHECK-CK860V: "-target-feature" "+hard-tp" "-target-feature" "+e1" "-target-feature" "+e2" "-target-feature" "+2e3"
// CHECK-CK860V: "-target-feature" "+mp" "-target-feature" "+3e3r1" "-target-feature" "+3e3r2" "-target-feature" "+3e3r3"
// CHECK-CK860V: "-target-feature" "+3e7" "-target-feature" "+mp1e2" "-target-feature" "+7e10" "-target-feature" "+10e60"