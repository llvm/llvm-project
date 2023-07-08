// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fclangir-build-deferred-threshold=0 %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

class String {
  char *storage{nullptr};
  long size;
  long capacity;

public:
  String() : size{0} {}
  String(int size) : size{size} {}
  String(const char *s) {}
};

void test() {
  String s1{};
  String s2{1};
  String s3{"abcdefghijklmnop"};
}

// CHECK-NOT: cir.func linkonce_odr @_ZN6StringC2Ev
// CHECK-NOT: cir.func linkonce_odr @_ZN6StringC2Ei
// CHECK-NOT: cir.func linkonce_odr @_ZN6StringC2EPKc
// CHECK-NOT: cir.func linkonce_odr @_ZN6StringC1EPKc

// CHECK: cir.func @_Z4testv()
// CHECK:   cir.call @_ZN6StringC1Ev(%0) : (!cir.ptr<!ty_22class2EString22>) -> ()