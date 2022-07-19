// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

#define __maybe_undef __attribute__((maybe_undef))

// CHECK:      define{{.*}} void @t1(i32 noundef [[TMP1:%.*]], i32 noundef [[TMP2:%.*]], i32 noundef [[TMP3:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[TMP4:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[TMP5:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[TMP6:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 [[TMP1:%.*]], i32* [[TMP4:%.*]], align 4
// CHECK-NEXT:   store i32 [[TMP2:%.*]], i32* [[TMP5:%.*]], align 4
// CHECK-NEXT:   store i32 [[TMP3:%.*]], i32* [[TMP6:%.*]], align 4
// CHECK-NEXT:   ret void

// CHECK:      define{{.*}} void @t2(i32 noundef [[TMP1:%.*]], i32 noundef [[TMP2:%.*]], i32 noundef [[TMP3:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[TMP4:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[TMP5:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[TMP6:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 [[TMP1:%.*]], i32* [[TMP4:%.*]], align 4
// CHECK-NEXT:   store i32 [[TMP2:%.*]], i32* [[TMP5:%.*]], align 4
// CHECK-NEXT:   store i32 [[TMP3:%.*]], i32* [[TMP6:%.*]], align 4
// CHECK-NEXT:   [[TMP7:%.*]] = load i32, i32* [[TMP4:%.*]], align 4
// CHECK-NEXT:   [[TMP8:%.*]] = load i32, i32* [[TMP5:%.*]], align 4
// CHECK-NEXT:   [[TMP9:%.*]] = load i32, i32* [[TMP6:%.*]], align 4
// CHECK-NEXT:   [[TMP10:%.*]] = freeze i32 [[TMP8:%.*]]
// CHECK-NEXT:   call void @t1(i32 noundef [[TMP7:%.*]], i32 noundef [[TMP10:%.*]], i32 noundef [[TMP9:%.*]])
// CHECK-NEXT:   ret void

void t1(int param1, int __maybe_undef param2, int param3) {}

void t2(int param1, int param2, int param3) {
    t1(param1, param2, param3);
}

// CHECK:      define{{.*}} void @TestVariadicFunction(i32 noundef [[TMP0:%.*]], ...)
// CHECK-NEXT: entry:
// CHECK-NEXT:  [[TMP1:%.*]] = alloca i32, align 4
// CHECK-NEXT:  [[TMP2:%.*]] = alloca i32, align 4
// CHECK-NEXT:  store i32 [[TMP0:%.*]], i32* [[TMP1:%.*]], align 4
// CHECK-NEXT:  [[TMP3:%.*]] = load i32, i32* [[TMP1:%.*]], align 4
// CHECK-NEXT:  [[TMP4:%.*]] = load i32, i32* [[TMP2:%.*]], align 4
// CHECK-NEXT:  [[TMP5:%.*]] = load i32, i32* [[TMP2:%.*]], align 4
// CHECK-NEXT:  [[TMP5:%.*]] = freeze i32 [[TMP2:%.*]]
// CHECK-NEXT:  call void (i32, ...) @VariadicFunction(i32 noundef [[TMP6:%.*]], i32 noundef [[TMP4:%.*]], i32 noundef [[TMP5:%.*]])
// CHECK-NEXT:  ret void

// CHECK: declare{{.*}} void @VariadicFunction(i32 noundef, ...)

void VariadicFunction(int __maybe_undef x, ...);
void TestVariadicFunction(int x, ...) {
  int Var;
  return VariadicFunction(x, Var, Var);
}

// CHECK:      define{{.*}} void @other()
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[TMP1:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[TMP2:%.*]] = load i32, i32* [[TMP1:%.*]], align 4
// CHECK-NEXT:   call void @func(i32 noundef [[TMP2:%.*]])
// CHECK-NEXT:   [[TMP3:%.*]] = load i32, i32* [[TMP1:%.*]], align 4
// CHECK-NEXT:   [[TMP4:%.*]] = freeze i32 [[TMP3:%.*]]
// CHECK-NEXT:   call void @func1(i32 noundef [[TMP4:%.*]])
// CHECK-NEXT:   ret void

// CHECK:      define{{.*}} void @func(i32 noundef [[TMP1:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[TMP2:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 [[TMP1:%.*]], i32* [[TMP2:%.*]], align 4
// CHECK-NEXT:   ret void

// CHECK:      define{{.*}} void @func1(i32 noundef [[TMP1:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[TMP2:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 [[TMP1:%.*]], i32* [[TMP2:%.*]], align 4
// CHECK-NEXT:   ret void

void func(int param);
void func1(int __maybe_undef param);

void other() {
  int Var;
  func(Var);
  func1(Var);
}

void func(__maybe_undef int param) {}
void func1(int param) {}

// CHECK:      define{{.*}} void @foo(i32 noundef [[TMP1:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[TMP2:%.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 [[TMP1:%.*]], i32* [[TMP2:%.*]], align 4
// CHECK-NEXT:   ret void

// CHECK:      define{{.*}} void @bar()
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[TMP1:%.*]] = alloca i32, align 4
// CHECK-NEXT:   [[TMP2:%.*]] = load i32, i32* [[TMP1:%.*]], align 4
// CHECK-NEXT:   call void @foo(i32 noundef [[TMP2:%.*]])
// CHECK-NEXT:   ret void

void foo(__maybe_undef int param);
void foo(int param) {}

void bar() {
  int Var;
  foo(Var);
}
