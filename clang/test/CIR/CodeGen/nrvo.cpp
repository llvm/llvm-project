// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -clangir-disable-emit-cxx-default -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

std::vector<const char*> test_nrvo() {
  std::vector<const char*> result;
  result.push_back("Words bend our thinking to infinite paths of self-delusion");
  return result;
}

// CHECK: !ty_22class2Estd3A3Avector22 = !cir.struct<"class.std::vector" {!cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!cir.ptr<!s8i>>}>

// CHECK: cir.func @_Z9test_nrvov() -> !ty_22class2Estd3A3Avector22
// CHECK:   %0 = cir.alloca !ty_22class2Estd3A3Avector22, cir.ptr <!ty_22class2Estd3A3Avector22>, ["__retval", init] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !cir.bool, cir.ptr <!cir.bool>, ["nrvo"] {alignment = 1 : i64}
// CHECK:   %2 = cir.const(#false) : !cir.bool
// CHECK:   cir.store %2, %1 : !cir.bool, cir.ptr <!cir.bool>
// CHECK:   cir.call @_ZNSt6vectorIPKcEC1Ev(%0) : (!cir.ptr<!ty_22class2Estd3A3Avector22>) -> ()
// CHECK:   cir.scope {
// CHECK:     %5 = cir.alloca !cir.ptr<!s8i>, cir.ptr <!cir.ptr<!s8i>>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK:     %6 = cir.get_global @".str" : cir.ptr <!cir.array<!s8i x 59>>
// CHECK:     %7 = cir.cast(array_to_ptrdecay, %6 : !cir.ptr<!cir.array<!s8i x 59>>), !cir.ptr<!s8i>
// CHECK:     cir.store %7, %5 : !cir.ptr<!s8i>, cir.ptr <!cir.ptr<!s8i>>
// CHECK:     cir.call @_ZNSt6vectorIPKcE9push_backEOS1_(%0, %5) : (!cir.ptr<!ty_22class2Estd3A3Avector22>, !cir.ptr<!cir.ptr<!s8i>>) -> ()
// CHECK:   }
// CHECK:   %3 = cir.const(#true) : !cir.bool
// CHECK:   cir.store %3, %1 : !cir.bool, cir.ptr <!cir.bool>
// CHECK:   %4 = cir.load %0 : cir.ptr <!ty_22class2Estd3A3Avector22>, !ty_22class2Estd3A3Avector22
// CHECK:   cir.return %4 : !ty_22class2Estd3A3Avector22
// CHECK: }
