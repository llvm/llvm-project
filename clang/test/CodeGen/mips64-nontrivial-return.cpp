// RUN: %clang_cc1 -triple mips64el-unknown-linux -O3 -target-abi n64 -o - -emit-llvm %s | FileCheck %s

class B {
public:
  virtual ~B() {}
};

class D : public B {
};

extern D gd0;

// CHECK: _Z4foo1v(%class.D* noalias nocapture writeonly sret

D foo1(void) {
  return gd0;
}
