// RUN: %clang_cc1 -triple=x86_64-pc-windows-msvc -debug-info-kind=limited \
// RUN:    -std=c++11 -gcodeview -emit-llvm -o - %s \
// RUN:    | FileCheck -check-prefix=NONEST %s
// RUN: %clang_cc1 -triple=x86_64-pc-windows-msvc -debug-info-kind=limited \
// RUN:    -std=c++11 -gcodeview -dwarf-column-info -emit-llvm -o - %s \
// RUN:    | FileCheck -check-prefix=COLUMNS %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -debug-info-kind=limited \
// RUN:    -std=c++11 -emit-llvm -o - %s | FileCheck -check-prefix=NESTED %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -debug-info-kind=limited \
// RUN:    -std=c++11 -dwarf-column-info -emit-llvm -o - %s \
// RUN:    | FileCheck -check-prefix=COLUMNS %s

class Foo {
public:
  static Foo create();
  void func();
  int *begin();
  int *end();
};

int bar(int x, int y);
int baz(int x, int y);
int qux(int x, int y);
int onearg(int x);
int noargs();
int noargs1();
Foo range(int x);

int foo(int x, int y, int z) {
  int a = bar(x, y) +
          baz(x, z) +
          qux(y, z);
  // NONEST: call i32 @{{.*}}bar{{.*}}, !dbg ![[LOC:[0-9]+]]
  // NONEST: call i32 @{{.*}}baz{{.*}}, !dbg ![[LOC]]
  // NONEST: call i32 @{{.*}}qux{{.*}}, !dbg ![[LOC]]
  // NONEST: store i32 {{.*}}, i32* %a,{{.*}} !dbg ![[LOC]]
  // NESTED: call i32 @{{.*}}bar{{.*}}, !dbg ![[BAR:[0-9]+]]
  // NESTED: call i32 @{{.*}}baz{{.*}}, !dbg ![[BAZ:[0-9]+]]
  // NESTED: call i32 @{{.*}}qux{{.*}}, !dbg ![[QUX:[0-9]+]]
  // NESTED: store i32 {{.*}}, i32* %a,{{.*}} !dbg ![[BAR]]
  // COLUMNS: call i32 @{{.*}}bar{{.*}}, !dbg ![[BAR:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}baz{{.*}}, !dbg ![[BAZ:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}qux{{.*}}, !dbg ![[QUX:[0-9]+]]
  // COLUMNS: store i32 {{.*}}, i32* %a,{{.*}} !dbg ![[DECLA:[0-9]+]]

  int i = 1, b = 0, c = 0;
  // NONEST: store i32 1, i32* %i,{{.*}} !dbg ![[ILOC:[0-9]+]]
  // NONEST: store i32 0, i32* %b,{{.*}} !dbg ![[ILOC]]
  // NONEST: store i32 0, i32* %c,{{.*}} !dbg ![[ILOC]]
  // NESTED: store i32 1, i32* %i,{{.*}} !dbg ![[ILOC:[0-9]+]]
  // NESTED: store i32 0, i32* %b,{{.*}} !dbg ![[ILOC]]
  // NESTED: store i32 0, i32* %c,{{.*}} !dbg ![[ILOC]]
  // COLUMNS: store i32 1, i32* %i,{{.*}} !dbg ![[ILOC:[0-9]+]]
  // COLUMNS: store i32 0, i32* %b,{{.*}} !dbg ![[BLOC:[0-9]+]]
  // COLUMNS: store i32 0, i32* %c,{{.*}} !dbg ![[CLOC:[0-9]+]]

  while (i > 0) {
    b = bar(a, b);
    --i;
  }
  // NONEST: call i32 @{{.*}}bar{{.*}}, !dbg ![[WHILE1:[0-9]+]]
  // NONEST: store i32 %{{[^,]+}}, i32* %i,{{.*}} !dbg ![[WHILE2:[0-9]+]]
  // NESTED: call i32 @{{.*}}bar{{.*}}, !dbg ![[WHILE1:[0-9]+]]
  // NESTED: store i32 %{{[^,]+}}, i32* %i,{{.*}} !dbg ![[WHILE2:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}bar{{.*}}, !dbg ![[WHILE1:[0-9]+]]
  // COLUMNS: store i32 %{{[^,]+}}, i32* %i,{{.*}} !dbg ![[WHILE2:[0-9]+]]

  for (i = 0; i < 1; i++) {
    b = bar(a, b);
    c = qux(a, c);
  }
  // NONEST: call i32 @{{.*}}bar{{.*}}, !dbg ![[FOR1:[0-9]+]]
  // NONEST: call i32 @{{.*}}qux{{.*}}, !dbg ![[FOR2:[0-9]+]]
  // NESTED: call i32 @{{.*}}bar{{.*}}, !dbg ![[FOR1:[0-9]+]]
  // NESTED: call i32 @{{.*}}qux{{.*}}, !dbg ![[FOR2:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}bar{{.*}}, !dbg ![[FOR1:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}qux{{.*}}, !dbg ![[FOR2:[0-9]+]]

  if (a < b) {
    int t = a;
    a = b;
    b = t;
  }
  // NONEST: store i32 %{{[^,]+}}, i32* %t,{{.*}} !dbg ![[IF1:[0-9]+]]
  // NONEST: store i32 %{{[^,]+}}, i32* %a,{{.*}} !dbg ![[IF2:[0-9]+]]
  // NONEST: store i32 %{{[^,]+}}, i32* %b,{{.*}} !dbg ![[IF3:[0-9]+]]
  // NESTED: store i32 %{{[^,]+}}, i32* %t,{{.*}} !dbg ![[IF1:[0-9]+]]
  // NESTED: store i32 %{{[^,]+}}, i32* %a,{{.*}} !dbg ![[IF2:[0-9]+]]
  // NESTED: store i32 %{{[^,]+}}, i32* %b,{{.*}} !dbg ![[IF3:[0-9]+]]
  // COLUMNS: store i32 %{{[^,]+}}, i32* %t,{{.*}} !dbg ![[IF1:[0-9]+]]
  // COLUMNS: store i32 %{{[^,]+}}, i32* %a,{{.*}} !dbg ![[IF2:[0-9]+]]
  // COLUMNS: store i32 %{{[^,]+}}, i32* %b,{{.*}} !dbg ![[IF3:[0-9]+]]

  int d = onearg(
      noargs());
  // NONEST: call i32 @{{.*}}noargs{{.*}}, !dbg ![[DECLD:[0-9]+]]
  // NONEST: call i32 @{{.*}}onearg{{.*}}, !dbg ![[DECLD]]
  // NONEST: store i32 %{{[^,]+}}, i32* %d,{{.*}} !dbg ![[DECLD]]
  // NESTED: call i32 @{{.*}}noargs{{.*}}, !dbg ![[DNOARGS:[0-9]+]]
  // NESTED: call i32 @{{.*}}onearg{{.*}}, !dbg ![[DECLD:[0-9]+]]
  // NESTED: store i32 %{{[^,]+}}, i32* %d,{{.*}} !dbg ![[DECLD]]
  // COLUMNS: call i32 @{{.*}}noargs{{.*}}, !dbg ![[DNOARGS:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}onearg{{.*}}, !dbg ![[DONEARG:[0-9]+]]
  // COLUMNS: store i32 %{{[^,]+}}, i32* %d,{{.*}} !dbg ![[DECLD:[0-9]+]]
  
  d = onearg(noargs());
  // NONEST: call i32 @{{.*}}noargs{{.*}}, !dbg ![[SETD:[0-9]+]]
  // NONEST: call i32 @{{.*}}onearg{{.*}}, !dbg ![[SETD]]
  // NONEST: store i32 %{{[^,]+}}, i32* %d,{{.*}} !dbg ![[SETD]]
  // NESTED: call i32 @{{.*}}noargs{{.*}}, !dbg ![[SETD:[0-9]+]]
  // NESTED: call i32 @{{.*}}onearg{{.*}}, !dbg ![[SETD]]
  // NESTED: store i32 %{{[^,]+}}, i32* %d,{{.*}} !dbg ![[SETD]]
  // COLUMNS: call i32 @{{.*}}noargs{{.*}}, !dbg ![[SETDNOARGS:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}onearg{{.*}}, !dbg ![[SETDONEARG:[0-9]+]]
  // COLUMNS: store i32 %{{[^,]+}}, i32* %d,{{.*}} !dbg ![[SETD:[0-9]+]]

  for (const auto x : range(noargs())) noargs1();
  // NONEST: call i32 @{{.*}}noargs{{.*}}, !dbg ![[RANGEFOR:[0-9]+]]
  // NONEST: call {{.+}} @{{.*}}range{{.*}}, !dbg ![[RANGEFOR]]
  // NONEST: call i32 @{{.*}}noargs1{{.*}}, !dbg ![[RANGEFOR_BODY:[0-9]+]]
  // NESTED: call i32 @{{.*}}noargs{{.*}}, !dbg ![[RANGEFOR:[0-9]+]]
  // NESTED: call {{.+}} @{{.*}}range{{.*}}, !dbg ![[RANGEFOR]]
  // NESTED: call i32 @{{.*}}noargs1{{.*}}, !dbg ![[RANGEFOR_BODY:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}noargs{{.*}}, !dbg ![[RANGEFOR_NOARGS:[0-9]+]]
  // COLUMNS: call {{.+}} @{{.*}}range{{.*}}, !dbg ![[RANGEFOR_RANGE:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}noargs1{{.*}}, !dbg ![[RANGEFOR_BODY:[0-9]+]]

  if (noargs() && noargs1()) {
    Foo::create().func();
  }
  // NONEST: call i32 @{{.*}}noargs{{.*}}, !dbg ![[AND:[0-9]+]]
  // NONEST: call i32 @{{.*}}noargs1{{.*}}, !dbg ![[AND]]
  // NONEST: call {{.+}} @{{.*}}create{{.*}}, !dbg ![[AND_BODY:[0-9]+]]
  // NONEST: call void @{{.*}}func{{.*}}, !dbg ![[AND_BODY]]
  // NESTED: call i32 @{{.*}}noargs{{.*}}, !dbg ![[AND:[0-9]+]]
  // NESTED: call i32 @{{.*}}noargs1{{.*}}, !dbg ![[AND]]
  // NESTED: call {{.+}} @{{.*}}create{{.*}}, !dbg ![[AND_BODY:[0-9]+]]
  // NESTED: call void @{{.*}}func{{.*}}, !dbg ![[AND_BODY]]
  // COLUMNS: call i32 @{{.*}}noargs{{.*}}, !dbg ![[ANDLHS:[0-9]+]]
  // COLUMNS: call i32 @{{.*}}noargs1{{.*}}, !dbg ![[ANDRHS:[0-9]+]]
  // COLUMNS: call {{.+}} @{{.*}}create{{.*}}, !dbg ![[AND_CREATE:[0-9]+]]
  // COLUMNS: call void @{{.*}}func{{.*}}, !dbg ![[AND_FUNC:[0-9]+]]

  return a -
         (b * z);
  // NONEST: mul nsw i32 {{.*}}, !dbg ![[RETLOC:[0-9]+]]
  // NONEST: sub nsw i32 {{.*}}, !dbg ![[RETLOC]]
  // NONEST: ret i32 {{.*}}, !dbg ![[RETLOC]]
  // NESTED: mul nsw i32 {{.*}}, !dbg ![[RETMUL:[0-9]+]]
  // NESTED: sub nsw i32 {{.*}}, !dbg ![[RETSUB:[0-9]+]]
  // NESTED: ret i32 {{.*}}, !dbg !
  // COLUMNS: mul nsw i32 {{.*}}, !dbg ![[RETMUL:[0-9]+]]
  // COLUMNS: sub nsw i32 {{.*}}, !dbg ![[RETSUB:[0-9]+]]
  // COLUMNS: ret i32 {{.*}}, !dbg !
}

// NONEST: ![[WHILE1]] = !DILocation(
// NONEST: ![[WHILE2]] = !DILocation(
// NONEST: ![[FOR1]] = !DILocation(
// NONEST: ![[FOR2]] = !DILocation(
// NONEST: ![[IF1]] = !DILocation(
// NONEST: ![[IF2]] = !DILocation(
// NONEST: ![[IF3]] = !DILocation(
// NONEST: ![[RANGEFOR]] = !DILocation(
// NONEST-SAME: line: [[RANGEFOR_LINE:[0-9]+]]
// NONEST: ![[RANGEFOR_BODY]] = !DILocation(
// NONEST-SAME: line: [[RANGEFOR_LINE]]

// NESTED: ![[BAR]] = !DILocation(
// NESTED: ![[BAZ]] = !DILocation(
// NESTED: ![[QUX]] = !DILocation(
// NESTED: ![[DECLD]] = !DILocation
// NESTED: ![[DNOARGS]] = !DILocation
// NESTED: ![[RANGEFOR]] = !DILocation(
// NESTED-SAME: line: [[RANGEFOR_LINE:[0-9]+]]
// NESTED: ![[RANGEFOR_BODY]] = !DILocation(
// NESTED-SAME: line: [[RANGEFOR_LINE]]
// NESTED: ![[RETSUB]] = !DILocation(
// NESTED: ![[RETMUL]] = !DILocation(

// COLUMNS: ![[DECLA]] = !DILocation(
// COLUMNS: ![[BAR]] = !DILocation(
// COLUMNS: ![[BAZ]] = !DILocation(
// COLUMNS: ![[QUX]] = !DILocation(
// COLUMNS: ![[ILOC]] = !DILocation(
// COLUMNS: ![[BLOC]] = !DILocation(
// COLUMNS: ![[CLOC]] = !DILocation(
// COLUMNS: ![[DECLD]] = !DILocation(
// COLUMNS: ![[DNOARGS]] = !DILocation(
// COLUMNS: ![[DONEARG]] = !DILocation(
// COLUMNS: ![[SETDNOARGS]] = !DILocation(
// COLUMNS: ![[SETDONEARG]] = !DILocation(
// COLUMNS: ![[SETD]] = !DILocation(
// COLUMNS: ![[RANGEFOR_NOARGS]] = !DILocation(
// COLUMNS: ![[RANGEFOR_RANGE]] = !DILocation(
// COLUMNS: ![[RANGEFOR_BODY]] = !DILocation(
// COLUMNS: ![[ANDLHS]] = !DILocation
// COLUMNS: ![[ANDRHS]] = !DILocation
// COLUMNS: ![[AND_CREATE]] = !DILocation
// COLUMNS: ![[AND_FUNC]] = !DILocation
// COLUNMS: ![[RETSUB]] = !DILocation(
// COLUMNS: ![[RETMUL]] = !DILocation(
