// RUN: %clang_cc1 -no-opaque-pointers -std=c89 -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C99 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c99 -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C99 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c11 -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C11 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c18 -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C11 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c2x -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C11 %s
//
// Check -ffinite-loops option in combination with various standard versions.
// RUN: %clang_cc1 -no-opaque-pointers -std=c89 -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=FINITE %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c99 -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=FINITE %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c11 -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=FINITE %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c18 -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=FINITE %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c2x -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=FINITE %s
//
// Check -fno-finite-loops option in combination with various standard versions.
// RUN: %clang_cc1 -no-opaque-pointers -std=c89 -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C99 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c99 -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C99 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c11 -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C99 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c18 -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C99 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c2x -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=C99 %s

int a = 0;
int b = 0;

// CHECK: datalayout
//
// CHECK-NOT: mustprogress
// CHECK-LABEL: @f0(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// C99-NOT:       br {{.*}}!llvm.loop
// C11-NOT:       br {{.*}}!llvm.loop
// FINITE-NEXT:   br {{.*}}!llvm.loop
//
void f0(void) {
  for (; ;) ;
}

// CHECK-NOT: mustprogress
// CHECK-LABEL: @f1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// CHECK-NEXT:    br i1 true, label %for.body, label %for.end
// CHECK:       for.body:
// C99-NOT:       br {{.*}}, !llvm.loop
// C11-NOT:       br {{.*}}, !llvm.loop
// FINITE-NEXT:   br {{.*}}, !llvm.loop
// CHECK:       for.end:
// CHECK-NEXT:    ret void
//
void f1(void) {
  for (; 1;) {
  }
}

// CHECK-NOT: mustprogress
// CHECK-LABEL: @f2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br i1 [[CMP]], label %for.body, label %for.end
// CHECK:       for.body:
// C99-NOT:       br {{.*}} !llvm.loop
// C11:           br label %for.cond, !llvm.loop [[LOOP1:!.*]]
// FINITE:        br label %for.cond, !llvm.loop [[LOOP1:!.*]]
// CHECK:       for.end:
// CHECK-NEXT:    ret void
//
void f2(void) {
  for (; a == b;) {
  }
}

// CHECK-NOT: mustprogress
// CHECK-LABEL: @F(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// CHECK-NEXT:    br i1 true, label %for.body, label %for.end
// CHECK:       for.body:
// C99-NOT:       br {{.*}}, !llvm.loop
// C11-NOT:       br {{.*}}, !llvm.loop
// FINITE-NEXT:   br {{.*}}, !llvm.loop
// CHECK:       for.end:
// CHECK-NEXT:    br label %for.cond1
// CHECK:       for.cond1:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br i1 [[CMP]], label %for.body2, label %for.end3
// CHECK:       for.body2:
// C99-NOT:       br {{.*}}, !llvm.loop
// C11:           br label %for.cond1, !llvm.loop [[LOOP2:!.*]]
// FINITE:        br label %for.cond1, !llvm.loop [[LOOP2:!.*]]
// CHECK:       for.end3:
// CHECK-NEXT:    ret void
//
void F(void) {
  for (; 1;) {
  }
  for (; a == b;) {
  }
}

// CHECK-NOT: mustprogress
// CHECK-LABEL: @w1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %while.body
// CHECK:       while.body:
// C99-NOT:       br {{.*}}, !llvm.loop
// C11-NOT:       br {{.*}}, !llvm.loop
// FINITE-NEXT:   br {{.*}}, !llvm.loop
//
void w1(void) {
  while (1) {
  }
}

// CHECK-NOT: mustprogress
// CHECK-LABEL: @w2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %while.cond
// CHECK:       while.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br i1 [[CMP]], label %while.body, label %while.end
// CHECK:       while.body:
// C99-NOT:       br {{.*}}, !llvm.loop
// C11:           br label %while.cond, !llvm.loop [[LOOP3:!.*]]
// FINITE:        br label %while.cond, !llvm.loop [[LOOP3:!.*]]
// CHECK:       while.end:
// CHECK-NEXT:    ret void
//
void w2(void) {
  while (a == b) {
  }
}

// CHECK-NOT: mustprogress
// CHECK-LABEL: @W(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label [[WHILE_COND:%.*]]
// CHECK:       while.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br i1 [[CMP]], label %while.body, label %while.end
// CHECK:       while.body:
// C99-NOT:       br {{.*}} !llvm.loop
// C11-NEXT:      br label %while.cond, !llvm.loop [[LOOP4:!.*]]
// FINITE-NEXT:   br label %while.cond, !llvm.loop [[LOOP4:!.*]]
// CHECK:       while.end:
// CHECK-NEXT:    br label %while.body2
// CHECK:       while.body2:
// C99-NOT:       br {{.*}} !llvm.loop
// C11-NOT:       br {{.*}} !llvm.loop
// FINITE-NEXT:   br {{.*}} !llvm.loop
//
void W(void) {
  while (a == b) {
  }
  while (1) {
  }
}

// CHECK-NOT: mustprogress
// CHECK-LABEL: @d1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %do.body
// CHECK:       do.body:
// CHECK-NEXT:    br label %do.cond
// CHECK:       do.cond:
// C99-NOT:       br {{.*}}, !llvm.loop
// C11-NOT:       br {{.*}}, !llvm.loop
// FINITE-NEXT:   br {{.*}}, !llvm.loop
// CHECK:       do.end:
// CHECK-NEXT:    ret void
//
void d1(void) {
  do {
  } while (1);
}

// CHECK-NOT: mustprogress
// CHECK-LABEL: @d2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %do.body
// CHECK:       do.body:
// CHECK-NEXT:    br label %do.cond
// CHECK:       do.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// C99-NOT:       br {{.*}}, !llvm.loop
// C11:           br i1 [[CMP]], label %do.body, label %do.end, !llvm.loop [[LOOP5:!.*]]
// FINITE:        br i1 [[CMP]], label %do.body, label %do.end, !llvm.loop [[LOOP5:!.*]]
// CHECK:       do.end:
// CHECK-NEXT:    ret void
//
void d2(void) {
  do {
  } while (a == b);
}

// CHECK-NOT: mustprogress
// CHECK-LABEL: @D(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %do.body
// CHECK:       do.body:
// CHECK-NEXT:    br label %do.cond
// CHECK:       do.cond:
// CHECK-NOT:     br label {{.*}}, !llvm.loop
// CHECK:       do.end:
// CHECK-NEXT:    br label %do.body1
// CHECK:       do.body1:
// CHECK-NEXT:    br label %do.cond2
// CHECK:       do.cond2:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// C99-NOT:       br {{.*}}, !llvm.loop
// C11:           br i1 [[CMP]], label %do.body1, label %do.end3, !llvm.loop [[LOOP6:!.*]]
// FINITE:        br i1 [[CMP]], label %do.body1, label %do.end3, !llvm.loop [[LOOP6:!.*]]
// CHECK:       do.end3:
// CHECK-NEXT:    ret void
//
void D(void) {
  do {
  } while (1);
  do {
  } while (a == b);
}

// C11: [[LOOP1]] = distinct !{[[LOOP1]], [[MP:!.*]]}
// C11: [[MP]] = !{!"llvm.loop.mustprogress"}
// C11: [[LOOP2]] = distinct !{[[LOOP2]], [[MP]]}
// C11: [[LOOP3]] = distinct !{[[LOOP3]], [[MP]]}
// C11: [[LOOP4]] = distinct !{[[LOOP4]], [[MP]]}
// C11: [[LOOP5]] = distinct !{[[LOOP5]], [[MP]]}
// C11: [[LOOP6]] = distinct !{[[LOOP6]], [[MP]]}
