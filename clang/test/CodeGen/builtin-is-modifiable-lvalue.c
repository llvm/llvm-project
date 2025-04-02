// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s| FileCheck %s

void report(int value);

__attribute__((__noinline__)) int passthru(int a)
{
  return a;
}

int global;
const int global_ro;

int checkme(int arg, const int arg_ro)
{
  int autovar = 7;

  // CHECK: call void @report(i32 noundef 0)
  report(__builtin_is_modifiable_lvalue(5));
  // CHECK: call void @report(i32 noundef 0)
  report(__builtin_is_modifiable_lvalue(checkme));
  // CHECK: call void @report(i32 noundef 1)
  report(__builtin_is_modifiable_lvalue(arg));
  // CHECK: call void @report(i32 noundef 0)
  report(__builtin_is_modifiable_lvalue(arg + 5));
  // CHECK: call void @report(i32 noundef 0)
  report(__builtin_is_modifiable_lvalue(arg_ro));
  // CHECK: call void @report(i32 noundef 1)
  report(__builtin_is_modifiable_lvalue(autovar));
  // CHECK: call void @report(i32 noundef 1)
  report(__builtin_is_modifiable_lvalue(global));
  // CHECK: call void @report(i32 noundef 0)
  report(__builtin_is_modifiable_lvalue(global_ro));
  // CHECK: call void @report(i32 noundef 1)
  report(__builtin_is_modifiable_lvalue((unsigned char)arg));
  // CHECK: call void @report(i32 noundef 0)
  report(__builtin_is_modifiable_lvalue(passthru(arg)));
  // CHECK: call void @report(i32 noundef 0)
  report(__builtin_is_modifiable_lvalue(""));
  // CHECK: load
  arg++;
  // CHECK: call void @report(i32 noundef 0)
  // CHECK-NOT: load
  report(__builtin_is_modifiable_lvalue(arg++));
  // CHECK: call void @report(i32 noundef 0)
  // CHECK-NOT: load
  report(__builtin_is_modifiable_lvalue(++arg));

  // CHECK: load
  return arg;
}
