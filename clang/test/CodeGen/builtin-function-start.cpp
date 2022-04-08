// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=cfi-icall -o - %s | FileCheck %s

#if !__has_builtin(__builtin_function_start)
#error "missing __builtin_function_start"
#endif

void a(void) {}
// CHECK: @e = global i8* bitcast (void ()* no_cfi @_Z1av to i8*)
const void *e = __builtin_function_start(a);

constexpr void (*d)() = &a;
// CHECK: @f = global i8* bitcast (void ()* no_cfi @_Z1av to i8*)
const void *f = __builtin_function_start(d);

void b(void) {}
// CHECK: @g = global [2 x i8*] [i8* bitcast (void ()* @_Z1bv to i8*), i8* bitcast (void ()* no_cfi @_Z1bv to i8*)]
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

// CHECK: define {{.*}}i32 @_ZNK1A1iEv(%class.A* {{.*}}%this)
int A::i() const { return 0; }

// CHECK: define {{.*}}i32 @_ZNK1A1iEi(%class.A* noundef {{.*}}%this, i32 noundef %n)
int A::i(int n) const { return 0; }

void h(void) {
  // CHECK: store i8* bitcast (void ()* no_cfi @_Z1bv to i8*), i8** %g
  void *g = __builtin_function_start(b);
  // CHECK: call void @_Z1cPv(i8* noundef bitcast (void ()* no_cfi @_Z1av to i8*))
  c(__builtin_function_start(a));

  // CHECK: store i8* bitcast (void (%class.A*)* no_cfi @_ZN1A1fEv to i8*), i8** %Af
  void *Af = __builtin_function_start(&A::f);
  // CHECK: store i8* bitcast (void (%class.A*)* no_cfi @_ZN1A1gEv to i8*), i8** %Ag
  void *Ag = __builtin_function_start(&A::g);
  // CHECK: store i8* bitcast (void ()* no_cfi @_ZN1A1hEv to i8*), i8** %Ah
  void *Ah = __builtin_function_start(&A::h);
  // CHECK: store i8* bitcast (i32 (%class.A*)* no_cfi @_ZNK1A1iEv to i8*), i8** %Ai1
  void *Ai1 = __builtin_function_start((int(A::*)() const) & A::i);
  // CHECK: store i8* bitcast (i32 (%class.A*, i32)* no_cfi @_ZNK1A1iEi to i8*), i8** %Ai2
  void *Ai2 = __builtin_function_start((int(A::*)(int) const) & A::i);
}
