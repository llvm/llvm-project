// RUN: %clang_cc1 %s -triple=arm-unknown-linux-gnueabi -target-abi aapcs -emit-llvm -o - | FileCheck %s

class SMLoc {
 const char *Ptr;
public:
 SMLoc();
 SMLoc(const SMLoc &RHS);
};
SMLoc foo(void *p);
void bar(void *x) {
 foo(x);
}
void zed(SMLoc x);
void baz() {
  SMLoc a;
  zed(a);
}

// CHECK: declare void @_Z3fooPv(ptr sret(%class.SMLoc) align 4, ptr noundef)
// CHECK: declare void @_Z3zed5SMLoc(ptr noundef)
