// RUN: %clang_cc1 %s -emit-llvm -o %t1
// RUN: FileCheck --check-prefix=FOO %s < %t1
// RUN: FileCheck --check-prefix=A %s < %t1
// RUN: FileCheck --check-prefix=BAR %s < %t1
// RUN: FileCheck --check-prefix=FOOS %s < %t1
// END.

static __attribute((annotate("sfoo_0"))) __attribute((annotate("sfoo_1"))) char sfoo;
__attribute((annotate("foo_0"))) __attribute((annotate("foo_1"))) char foo;

void __attribute((annotate("ann_a_0"))) __attribute((annotate("ann_a_1"))) __attribute((annotate("ann_a_2"))) __attribute((annotate("ann_a_3"))) a(char *a);
void __attribute((annotate("ann_a_0"))) __attribute((annotate("ann_a_1"))) a(char *a) {
  __attribute__((annotate("bar_0"))) __attribute__((annotate("bar_1"))) static char bar;
  sfoo = 0;
}

// FOOS: target triple
// FOOS: private unnamed_addr constant [7 x i8] c"sfoo_{{.}}\00", section "llvm.metadata"
// FOOS: private unnamed_addr constant [7 x i8] c"sfoo_{{.}}\00", section "llvm.metadata"
// FOOS-NOT: sfoo_
// FOOS: @llvm.global.annotations = appending global [10 x { i8*, i8*, i8*, i32 }] {{.*}}i8* @sfoo{{.*}}i8* @sfoo{{.*}}, section "llvm.metadata"

// FOO: target triple
// FOO: private unnamed_addr constant [6 x i8] c"foo_{{.}}\00", section "llvm.metadata"
// FOO: private unnamed_addr constant [6 x i8] c"foo_{{.}}\00", section "llvm.metadata"
// FOO-NOT: foo_
// FOO: @llvm.global.annotations = appending global [10 x { i8*, i8*, i8*, i32 }] {{.*}}i8* @foo{{.*}}i8* @foo{{.*}}, section "llvm.metadata"

// A: target triple
// A: private unnamed_addr constant [8 x i8] c"ann_a_{{.}}\00", section "llvm.metadata"
// A: private unnamed_addr constant [8 x i8] c"ann_a_{{.}}\00", section "llvm.metadata"
// A: private unnamed_addr constant [8 x i8] c"ann_a_{{.}}\00", section "llvm.metadata"
// A: private unnamed_addr constant [8 x i8] c"ann_a_{{.}}\00", section "llvm.metadata"
// A-NOT: ann_a_
// A: @llvm.global.annotations = appending global [10 x { i8*, i8*, i8*, i32 }] {{.*}}i8* bitcast (void (i8*)* @a to i8*){{.*}}i8* bitcast (void (i8*)* @a to i8*){{.*}}i8* bitcast (void (i8*)* @a to i8*){{.*}}i8* bitcast (void (i8*)* @a to i8*){{.*}}, section "llvm.metadata"

// BAR: target triple
// BAR: private unnamed_addr constant [6 x i8] c"bar_{{.}}\00", section "llvm.metadata"
// BAR: private unnamed_addr constant [6 x i8] c"bar_{{.}}\00", section "llvm.metadata"
// BAR-NOT: bar_
// BAR: @llvm.global.annotations = appending global [10 x { i8*, i8*, i8*, i32 }] {{.*}}i8* @a.bar{{.*}}i8* @a.bar{{.*}}, section "llvm.metadata"
