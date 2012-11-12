// RUN: %clang -target mips64el-unknown-linux -O3 -S -mabi=n64 -o - -emit-llvm %s | FileCheck %s

class B {
public:
  virtual ~B() {}
};

class D : public B {
};

extern D gd0;

// CHECK: _Z4foo1v(%class.D* noalias nocapture sret

D foo1(void) {
  return gd0;
}
