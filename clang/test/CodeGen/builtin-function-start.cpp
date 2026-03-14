// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=cfi-icall -o - %s | FileCheck %s

#if !__has_builtin(__builtin_function_start)
#error "missing __builtin_function_start"
#endif

void a(void) {}
// CHECK: @e = global ptr no_cfi @_Z1av
const void *e = __builtin_function_start(a);

constexpr void (*d)() = &a;
// CHECK: @f = global ptr no_cfi @_Z1av
const void *f = __builtin_function_start(d);

void b(void) {}
// CHECK: @g = global [2 x ptr] [ptr @_Z1bv, ptr no_cfi @_Z1bv]
void *g[] = {(void *)b, __builtin_function_start(b)};

void c(void *p) {}

class A {
public:
  void f();
  virtual void g();
  static void h();
  int i() const;
  int i(int n) const;
};

void A::f() {}
void A::g() {}
void A::h() {}

// CHECK: define {{.*}}i32 @_ZNK1A1iEv(ptr {{.*}}%this)
int A::i() const { return 0; }

// CHECK: define {{.*}}i32 @_ZNK1A1iEi(ptr noundef {{.*}}%this, i32 noundef %n)
int A::i(int n) const { return 0; }

void h(void) {
  // CHECK: store ptr no_cfi @_Z1bv, ptr %g
  void *g = __builtin_function_start(b);
  // CHECK: call void @_Z1cPv(ptr noundef no_cfi @_Z1av)
  c(__builtin_function_start(a));

  // CHECK: store ptr no_cfi @_ZN1A1fEv, ptr %Af
  void *Af = __builtin_function_start(&A::f);
  // CHECK: store ptr no_cfi @_ZN1A1gEv, ptr %Ag
  void *Ag = __builtin_function_start(&A::g);
  // CHECK: store ptr no_cfi @_ZN1A1hEv, ptr %Ah
  void *Ah = __builtin_function_start(&A::h);
  // CHECK: store ptr no_cfi @_ZNK1A1iEv, ptr %Ai1
  void *Ai1 = __builtin_function_start((int(A::*)() const) & A::i);
  // CHECK: store ptr no_cfi @_ZNK1A1iEi, ptr %Ai2
  void *Ai2 = __builtin_function_start((int(A::*)(int) const) & A::i);
}
