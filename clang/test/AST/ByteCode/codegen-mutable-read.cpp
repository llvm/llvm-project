// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s                                         | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s


/// In the if expression below, the read from s.i should fail.
/// If it doesn't, and we actually read the value 0, the call to
/// func() will always occur, resuliting in a runtime failure.

struct S {
  mutable int i = 0;
};

void func() {
  __builtin_abort();
};

void setI(const S &s) {
  s.i = 12;
}

int main() {
  const S s;

  setI(s);

  if (s.i == 0)
    func();

  return 0;
}

// CHECK: define dso_local noundef i32 @main()
// CHECK: br
// CHECK: if.then
// CHECK: if.end
// CHECK: ret i32 0
