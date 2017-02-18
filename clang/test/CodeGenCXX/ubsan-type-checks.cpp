// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=alignment | FileCheck %s -check-prefixes=ALIGN,COMMON
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=null | FileCheck %s -check-prefixes=NULL,COMMON
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=object-size | FileCheck %s -check-prefixes=OBJSIZE,COMMON

struct A {
  // COMMON-LABEL: define linkonce_odr void @_ZN1A10do_nothingEv
  void do_nothing() {
    // ALIGN-NOT: ptrtoint %struct.A* %{{.*}} to i64, !nosanitize
    // ALIGN-NOT: and i64 %{{.*}}, 7, !nosanitize
 
    // NULL: icmp ne %struct.A* %{{.*}}, null, !nosanitize
 
    // OBJSIZE-NOT: call i64 @llvm.objectsize
  }
};

void force_irgen() {
  A a;
  a.do_nothing();
}
