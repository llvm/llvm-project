// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=SAMPLEPGO
// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -flto=thin -o - 2>&1 | FileCheck %s -check-prefix=THINLTO
// RUN: %clang_cc1 -O2 -fexperimental-new-pass-manager -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=SAMPLEPGO
// FIXME: Run the following command once LTOPreLinkDefaultPipeline is
//        customized.
// %clang_cc1 -O2 -fexperimental-new-pass-manager -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -flto=thin -o - 2>&1 | FileCheck %s -check-prefix=THINLTO
// Checks if hot call is inlined by normal compile, but not inlined by
// thinlto compile.

int baz(int);
int g;

void foo(int n) {
  for (int i = 0; i < n; i++)
    g += baz(i);
}

// SAMPLEPGO-LABEL: define void @bar
// THINLTO-LABEL: define void @bar
// SAMPLEPGO-NOT: call{{.*}}foo
// THINLTO: call{{.*}}foo
void bar(int n) {
  for (int i = 0; i < n; i++)
    foo(i);
}

// Checks if loop unroll is invoked by normal compile, but not thinlto compile.
// SAMPLEPGO-LABEL: define void @unroll
// THINLTO-LABEL: define void @unroll
// SAMPLEPGO: call{{.*}}baz
// SAMPLEPGO: call{{.*}}baz
// THINLTO: call{{.*}}baz
// THINLTO-NOT: call{{.*}}baz
void unroll() {
  for (int i = 0; i < 2; i++)
    baz(i);
}

// Checks that icp is not invoked for ThinLTO, but invoked for normal samplepgo.
// SAMPLEPGO-LABEL: define void @icp
// THINLTO-LABEL: define void @icp
// SAMPLEPGO: if.true.direct_targ
// FIXME: the following condition needs to be reversed once
//        LTOPreLinkDefaultPipeline is customized.
// THINLTO-NOT: if.true.direct_targ
void icp(void (*p)()) {
  p();
}
