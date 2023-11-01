// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -clangir-disable-emit-cxx-default -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

namespace std {
  template<typename T>
  void vector<T>::resize(size_type __sz) {
    size_type __cs = size();
    if (__cs) {}
  }
} // namespace std

// CHECK: cir.func linkonce_odr @_ZNSt6vectorIyE6resizeEm(
// CHECK:   %0 = cir.alloca !cir.ptr<!ty_22std3A3Avector3Cunsigned_long_long3E22>, cir.ptr <!cir.ptr<!ty_22std3A3Avector3Cunsigned_long_long3E22>>, ["this", init] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !u64i, cir.ptr <!u64i>, ["__sz", init] {alignment = 8 : i64}
// CHECK:   %2 = cir.alloca !u64i, cir.ptr <!u64i>, ["__cs", init] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_22std3A3Avector3Cunsigned_long_long3E22>, cir.ptr <!cir.ptr<!ty_22std3A3Avector3Cunsigned_long_long3E22>>
// CHECK:   cir.store %arg1, %1 : !u64i, cir.ptr <!u64i>
// CHECK:   %3 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22std3A3Avector3Cunsigned_long_long3E22>>, !cir.ptr<!ty_22std3A3Avector3Cunsigned_long_long3E22>
// CHECK:   %4 = cir.call @_ZNKSt6vectorIyE4sizeEv(%3) : (!cir.ptr<!ty_22std3A3Avector3Cunsigned_long_long3E22>) -> !u64i
// CHECK:   cir.store %4, %2 : !u64i, cir.ptr <!u64i>
// CHECK:   cir.scope {
// CHECK:     %5 = cir.load %2 : cir.ptr <!u64i>, !u64i
// CHECK:     %6 = cir.cast(int_to_bool, %5 : !u64i), !cir.bool
// CHECK:     cir.if %6 {
// CHECK:     }
// CHECK:   }
// CHECK:   cir.return

void m() {
  std::vector<unsigned long long> a;
  int i = 43;
  a.resize(i);
}