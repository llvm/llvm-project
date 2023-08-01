// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -clangir-disable-emit-cxx-default -fclangir-lifetime-check="history=all;remarks=all" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

struct A {
  void* ctx;
  void setInfo(void** ctxPtr);
};

void A::setInfo(void** ctxPtr) {
  if (ctxPtr != nullptr) {
    *ctxPtr = ctx; // expected-remark {{pset => { fn_arg:1 }}}
  }
}