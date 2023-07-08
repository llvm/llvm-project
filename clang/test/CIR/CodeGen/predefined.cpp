// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

extern "C" {
  void __assert2(const char* __file, int __line, const char* __function, const char* __msg) __attribute__((__noreturn__));
}

void m() {
  __assert2("yo.cpp", 79, __PRETTY_FUNCTION__, "doom");
}

// CHECK: cir.func @_Z1mv()
// CHECK:     %0 = cir.get_global @".str" : cir.ptr <!cir.array<!s8i x 7>>
// CHECK:     %1 = cir.cast(array_to_ptrdecay, %0 : !cir.ptr<!cir.array<!s8i x 7>>), !cir.ptr<!s8i>
// CHECK:     %2 = cir.const(#cir.int<79> : !s32i) : !s32i
// CHECK:     %3 = cir.get_global @".str1" : cir.ptr <!cir.array<!s8i x 9>>
// CHECK:     %4 = cir.cast(array_to_ptrdecay, %3 : !cir.ptr<!cir.array<!s8i x 9>>), !cir.ptr<!s8i>
// CHECK:     %5 = cir.get_global @".str2" : cir.ptr <!cir.array<!s8i x 5>>
// CHECK:     %6 = cir.cast(array_to_ptrdecay, %5 : !cir.ptr<!cir.array<!s8i x 5>>), !cir.ptr<!s8i>
// CHECK:     cir.call @__assert2(%1, %2, %4, %6) : (!cir.ptr<!s8i>, !s32i, !cir.ptr<!s8i>, !cir.ptr<!s8i>) -> ()
// CHECK:     cir.return
// CHECK:   }
