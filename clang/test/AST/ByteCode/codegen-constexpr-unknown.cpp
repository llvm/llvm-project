// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -fcxx-exceptions -o - %s                                         | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -fcxx-exceptions -o - %s -fexperimental-new-constant-interpreter | FileCheck %s


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


/// Similarly, here we revisit the BindingDecl.
struct F { int x; };
int main2() {
  const F const s{99};
  const auto& [r1] = s;
  if (&r1 != &s.x)
    __builtin_abort();
  return 0;
}
// CHECK: define dso_local noundef i32 @_Z5main2v()
// CHECK: br
// CHECK: if.then
// CHECK: if.end
// CHECK: ret i32 0

/// The comparison here should work and return 0.
class X {
public:
  X();
  X(const X&);
  X(const volatile X &);
  ~X();
};
extern X OuterX;
X test24() {
  X x;
  if (&x == &OuterX)
    throw 0;
  return x;
}

// CHECK: define dso_local void @_Z6test24v
// CHECK-NOT: lpad
// CHECK-NOT: eh.resume
