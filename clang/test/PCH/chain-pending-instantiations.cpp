// RUN: %clang_cc1 %s -emit-llvm -triple i686-pc-linux -o - -chain-include %s -chain-include %s | FileCheck %s
// CHECK: define linkonce_odr noundef %{{[^ ]+}} @_ZN1AI1BE3getEv
#if !defined(PASS1)
#define PASS1

template <typename Derived>
struct A {
  Derived* get() { return 0; }
};

struct B : A<B> {
};

#elif !defined(PASS2)
#define PASS2

struct C : B {
};

struct D : C {
  void run() {
    (void)get();
  }
};

#else

int main() {
  D d;
  d.run();
}

#endif
