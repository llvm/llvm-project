// RUN: %clang_cc1 -fexperimental-new-pass-manager -fdebug-pass-manager -O2 -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=SAMPLEPGO
// RUN: %clang_cc1 -fexperimental-new-pass-manager -fdebug-pass-manager -O2 -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -flto=thin -o - 2>&1 | FileCheck %s -check-prefix=THINLTO

int baz(int);
int g;

void foo(int n) {
  for (int i = 0; i < n; i++)
    g += baz(i);
}

// Checks that loop unroll and icp are invoked by normal compile, but not thinlto compile.

// SAMPLEPGO:               Running pass: PGOIndirectCallPromotion on [module]
// SAMPLEPGO:               Running pass: LoopUnrollPass on bar

// SAMPLEPGO-OLDPM:         PGOIndirectCallPromotion
// SAMPLEPGO-OLDPM:         Unroll loops
// SAMPLEPGO-OLDPM:         Unroll loops

// THINLTO-NOT:             Running pass: PGOIndirectCallPromotion on [module]
// THINLTO-NOT:             Running pass: LoopUnrollPass on bar

// THINLTO-OLDPM-NOT:       PGOIndirectCallPromotion
// The first Unroll loop pass is the createSimpleLoopUnrollPass that unrolls and peels
// loops with small constant trip counts. The second one is skipped by ThinLTO.
// THINLTO-OLDPM:           Unroll loops
// THINLTO-OLDPM-NOT:       Unroll loops


// Checks if hot call is inlined by normal compile, but not inlined by
// thinlto compile.
// SAMPLEPGO-LABEL: define {{(dso_local )?}}void @bar
// THINLTO-LABEL: define {{(dso_local )?}}void @bar
// SAMPLEPGO-NOT: call{{.*}}foo
// THINLTO: call{{.*}}foo
void bar(int n) {
  for (int i = 0; i < n; i++)
    foo(i);
}
