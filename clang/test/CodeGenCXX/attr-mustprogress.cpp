// RUN: %clang_cc1 -std=c++98 -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX98 %s
// RUN: %clang_cc1 -std=c++11 -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX11 %s
// RUN: %clang_cc1 -std=c++14 -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX11 %s
// RUN: %clang_cc1 -std=c++17 -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX11 %s
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX11 %s

// Check -ffinite-loops option in combination with various standard versions.
// RUN: %clang_cc1 -std=c++98 -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=FINITE %s
// RUN: %clang_cc1 -std=c++11 -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX11 %s
// RUN: %clang_cc1 -std=c++14 -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX11 %s
// RUN: %clang_cc1 -std=c++17 -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX11 %s
// RUN: %clang_cc1 -std=c++20 -ffinite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX11 %s

// Check -fno-finite-loops option in combination with various standard versions.
// RUN: %clang_cc1 -std=c++98 -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX98 %s
// RUN: %clang_cc1 -std=c++11 -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX98 %s
// RUN: %clang_cc1 -std=c++14 -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX98 %s
// RUN: %clang_cc1 -std=c++17 -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX98 %s
// RUN: %clang_cc1 -std=c++20 -fno-finite-loops -triple=x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=CXX98 %s

int a = 0;
int b = 0;

// CHECK: datalayout

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2f0v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// CXX98-NOT:    br {{.*}} llvm.loop
// CXX11-NEXT:   br label %for.cond, !llvm.loop [[LOOP1:!.*]]
// FINITE-NEXT:  br label %for.cond, !llvm.loop [[LOOP1:!.*]]
void f0() {
  for (; ;) ;
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2f1v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// CHECK-NEXT:    br i1 true, label %for.body, label %for.end
// CHECK:       for.body:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br label %for.cond, !llvm.loop [[LOOP2:!.*]]
// FINITE-NEXT:  br label %for.cond, !llvm.loop [[LOOP2:!.*]]
// CHECK:       for.end:
// CHECK-NEXT:    ret void
//
void f1() {
  for (; 1;)
    ;
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2f2v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br i1 [[CMP]], label %for.body, label %for.end
// CHECK:       for.body:
// CXX98-NOT:    br {{.*}}, !llvm.loop
// CXX11:        br label %for.cond, !llvm.loop [[LOOP3:!.*]]
// FINITE-NEXT:  br label %for.cond, !llvm.loop [[LOOP3:!.*]]
// CHECK:       for.end:
// CHECK-NEXT:    ret void
//
void f2() {
  for (; a == b;)
    ;
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z1Fv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// CHECK-NEXT:    br i1 true, label %for.body, label %for.end
// CHECK:       for.body:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br label %for.cond, !llvm.loop [[LOOP4:!.*]]
// FINITE-NEXT:   br label %for.cond, !llvm.loop [[LOOP4:!.*]]
// CHECK:       for.end:
// CHECK-NEXT:    br label %for.cond1
// CHECK:       for.cond1:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br i1 [[CMP]], label %for.body2, label %for.end3
// CHECK:       for.body2:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br label %for.cond1, !llvm.loop [[LOOP5:!.*]]
// FINITE-NEXT:   br label %for.cond1, !llvm.loop [[LOOP5:!.*]]
// CHECK:       for.end3:
// CHECK-NEXT:    ret void
//
void F() {
  for (; 1;)
    ;
  for (; a == b;)
    ;
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2F2v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br i1 [[CMP]], label %for.body, label %for.end
// CHECK:       for.body:
// CXX98-NOT:     br {{.*}} !llvm.loop
// CXX11-NEXT:    br label %for.cond, !llvm.loop [[LOOP6:!.*]]
// FINITE-NEXT:   br label %for.cond, !llvm.loop [[LOOP6:!.*]]
// CHECK:       for.end:
// CHECK-NEXT:    br label %for.cond1
// CHECK:       for.cond1:
// CHECK-NEXT:    br i1 true, label %for.body2, label %for.end3
// CHECK:       for.body2:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br label %for.cond1, !llvm.loop [[LOOP7:!.*]]
// FINITE-NEXT:   br label %for.cond1, !llvm.loop [[LOOP7:!.*]]
// CHECK:       for.end3:
// CHECK-NEXT:    ret void
//
void F2() {
  for (; a == b;)
    ;
  for (; 1;)
    ;
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2w1v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %while.body
// CHECK:       while.body:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br label %while.body, !llvm.loop [[LOOP8:!.*]]
// FINITE-NEXT:   br label %while.body, !llvm.loop [[LOOP8:!.*]]
//
void w1() {
  while (1)
    ;
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2w2v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %while.cond
// CHECK:       while.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br i1 [[CMP]], label %while.body, label %while.end
// CHECK:       while.body:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br label %while.cond, !llvm.loop [[LOOP9:!.*]]
// FINITE-NEXT:   br label %while.cond, !llvm.loop [[LOOP9:!.*]]
// CHECK:       while.end:
// CHECK-NEXT:    ret void
//
void w2() {
  while (a == b)
    ;
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z1Wv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %while.cond
// CHECK:       while.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br i1 [[CMP]], label %while.body, label %while.end
// CHECK:       while.body:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br label %while.cond, !llvm.loop [[LOOP10:!.*]]
// FINITE-NEXT:   br label %while.cond, !llvm.loop [[LOOP10:!.*]]
// CHECK:       while.end:
// CHECK-NEXT:    br label %while.body2
// CHECK:       while.body2:
// CXX98-NOT:    br {{.*}}, !llvm.loop
// CXX11-NEXT:   br label %while.body2, !llvm.loop [[LOOP11:!.*]]
// FINITE-NEXT:  br label %while.body2, !llvm.loop [[LOOP11:!.*]]
//
void W() {
  while (a == b)
    ;
  while (1)
    ;
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2W2v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %while.body
// CHECK:       while.body:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br label %while.body, !llvm.loop [[LOOP12:!.*]]
// FINITE-NEXT:   br label %while.body, !llvm.loop [[LOOP12:!.*]]
//
void W2() {
  while (1)
    ;
  while (a == b)
    ;
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2d1v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %do.body
// CHECK:       do.body:
// CHECK-NEXT:    br label %do.cond
// CHECK:       do.cond:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br i1 true, label %do.body, label %do.end, !llvm.loop [[LOOP13:!.*]]
// FINITE-NEXT:   br i1 true, label %do.body, label %do.end, !llvm.loop [[LOOP13:!.*]]
// CHECK:       do.end:
// CHECK-NEXT:    ret void
//
void d1() {
  do
    ;
  while (1);
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2d2v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %do.body
// CHECK:       do.body:
// CHECK-NEXT:    br label %do.cond
// CHECK:       do.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br i1 [[CMP]], label %do.body, label %do.end, !llvm.loop [[LOOP14:!.*]]
// FINITE-NEXT:   br i1 [[CMP]], label %do.body, label %do.end, !llvm.loop [[LOOP14:!.*]]
// CHECK:       do.end:
// CHECK-NEXT:    ret void
//
void d2() {
  do
    ;
  while (a == b);
}

// CXX98-NOT:  mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z1Dv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %do.body
// CHECK:       do.body:
// CHECK-NEXT:    br label %do.cond
// CHECK:       do.cond:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br i1 true, label %do.body, label %do.end, !llvm.loop [[LOOP15:!.*]]
// FINITE-NEXT:   br i1 true, label %do.body, label %do.end, !llvm.loop [[LOOP15:!.*]]
// CHECK:       do.end:
// CHECK-NEXT:    br label %do.body1
// CHECK:       do.body1:
// CHECK-NEXT:    br label %do.cond2
// CHECK:       do.cond2:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br i1 [[CMP]], label %do.body1, label %do.end3, !llvm.loop [[LOOP16:!.*]]
// FINITE-NEXT:   br i1 [[CMP]], label %do.body1, label %do.end3, !llvm.loop [[LOOP16:!.*]]
// CHECK:       do.end3:
// CHECK-NEXT:    ret void
//
void D() {
  do
    ;
  while (1);
  do
    ;
  while (a == b);
}

// CXX98-NOT : mustprogress
// CXX11:      mustprogress
// FINITE-NOT: mustprogress
// CHECK-LABEL: @_Z2D2v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    br label %do.body
// CHECK:       do.body:
// CHECK-NEXT:    br label %do.cond
// CHECK:       do.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr @b, align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP0]], [[TMP1]]
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br i1 [[CMP]], label %do.body, label %do.end, !llvm.loop [[LOOP17:!.*]]
// FINITE-NEXT:   br i1 [[CMP]], label %do.body, label %do.end, !llvm.loop [[LOOP17:!.*]]
// CHECK:       do.end:
// CHECK-NEXT:    br label %do.body1
// CHECK:       do.body1:
// CHECK-NEXT:    br label %do.cond2
// CHECK:       do.cond2:
// CXX98-NOT:     br {{.*}}, !llvm.loop
// CXX11-NEXT:    br i1 true, label %do.body1, label %do.end3, !llvm.loop [[LOOP18:!.*]]
// FINITE-NEXT:   br i1 true, label %do.body1, label %do.end3, !llvm.loop [[LOOP18:!.*]]
// CHECK:       do.end3:
// CHECK-NEXT:    ret void
//
void D2() {
  do
    ;
  while (a == b);
  do
    ;
  while (1);
}

// CXX11: [[LOOP1]] = distinct !{[[LOOP1]], [[MP:!.*]]}
// CXX11: [[MP]] = !{!"llvm.loop.mustprogress"}
// CXX11: [[LOOP2]] = distinct !{[[LOOP2]], [[MP]]}
// CXX11: [[LOOP3]] = distinct !{[[LOOP3]], [[MP]]}
// CXX11: [[LOOP4]] = distinct !{[[LOOP4]], [[MP]]}
// CXX11: [[LOOP5]] = distinct !{[[LOOP5]], [[MP]]}
// CXX11: [[LOOP6]] = distinct !{[[LOOP6]], [[MP]]}
// CXX11: [[LOOP7]] = distinct !{[[LOOP7]], [[MP]]}
// CXX11: [[LOOP8]] = distinct !{[[LOOP8]], [[MP]]}
// CXX11: [[LOOP9]] = distinct !{[[LOOP9]], [[MP]]}
// CXX11: [[LOOP10]] = distinct !{[[LOOP10]], [[MP]]}
// CXX11: [[LOOP11]] = distinct !{[[LOOP11]], [[MP]]}
// CXX11: [[LOOP12]] = distinct !{[[LOOP12]], [[MP]]}
// CXX11: [[LOOP13]] = distinct !{[[LOOP13]], [[MP]]}
// CXX11: [[LOOP14]] = distinct !{[[LOOP14]], [[MP]]}
// CXX11: [[LOOP15]] = distinct !{[[LOOP15]], [[MP]]}
// CXX11: [[LOOP16]] = distinct !{[[LOOP16]], [[MP]]}
// CXX11: [[LOOP17]] = distinct !{[[LOOP17]], [[MP]]}
// CXX11: [[LOOP18]] = distinct !{[[LOOP18]], [[MP]]}
