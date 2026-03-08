// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -fclangir -emit-cir -fcxx-exceptions -fexceptions -mmlir --mlir-print-ir-before=cir-cxxabi-lowering -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR,CIR-BEFORE
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR,CIR-AFTER
// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -fclangir -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s --check-prefixes=LLVM
#include <typeinfo>

namespace Test1 {

// PR7400
struct A { virtual void f(); };

// CIR: cir.global constant external @_ZN5Test16int_tiE = #cir.global_view<@_ZTIi> : !cir.ptr<!rec_std3A3Atype_info>
// LLVM: @_ZN5Test16int_tiE ={{.*}} constant ptr @_ZTIi, align 8
const std::type_info &int_ti = typeid(int);

// CIR: cir.global constant external @_ZN5Test14A_tiE = #cir.global_view<@_ZTIN5Test11AE> : !cir.ptr<!rec_std3A3Atype_info>
// LLVM: @_ZN5Test14A_tiE ={{.*}} constant ptr @_ZTIN5Test11AE, align 8
const std::type_info &A_ti = typeid(const volatile A &);

volatile char c;

// CIR: cir.global constant external @_ZN5Test14c_tiE = #cir.global_view<@_ZTIc> : !cir.ptr<!rec_std3A3Atype_info>
// LLVM: @_ZN5Test14c_tiE ={{.*}} constant ptr @_ZTIc, align 8
const std::type_info &c_ti = typeid(c);

extern const double &d;

// CIR: cir.global constant external @_ZN5Test14d_tiE = #cir.global_view<@_ZTId> : !cir.ptr<!rec_std3A3Atype_info>
// LLVM: @_ZN5Test14d_tiE ={{.*}} constant ptr @_ZTId, align 8
const std::type_info &d_ti = typeid(d);

extern A &a;

// CIR-AFTER: cir.global external @_ZN5Test14a_tiE = #cir.ptr<null> : !cir.ptr<!rec_std3A3Atype_info>

// CIR-BEFORE: cir.global external @_ZN5Test14a_tiE = ctor : !cir.ptr<!rec_std3A3Atype_info> {
// CIR-AFTER: cir.func{{.*}}@__cxx_global_var_init() {
//
// CIR-NEXT: %[[GET_GLOB_ATI:.*]] = cir.get_global @_ZN5Test14a_tiE : !cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>
// CIR-NEXT: %[[GET_GLOB_A:.*]] = cir.get_global @_ZN5Test11aE : !cir.ptr<!cir.ptr<!rec_Test13A3AA>>
// CIR-NEXT: %[[LOAD_GLOB_A:.*]] = cir.load %[[GET_GLOB_A]] : !cir.ptr<!cir.ptr<!rec_Test13A3AA>>, !cir.ptr<!rec_Test13A3AA>
// CIR-NEXT: %[[GET_VPTR:.*]] = cir.vtable.get_vptr %[[LOAD_GLOB_A]] : !cir.ptr<!rec_Test13A3AA> -> !cir.ptr<!cir.vptr>
// CIR-NEXT: %[[LOAD_VPTR:.*]] = cir.load align(8) %[[GET_VPTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-BEFORE: %[[GET_TYPEINFO:.*]] = cir.vtable.get_type_info %[[LOAD_VPTR]] : !cir.vptr -> !cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>
//
// CIR-AFTER: %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CIR-AFTER: %[[VPTR_CAST:.*]] = cir.cast bitcast %[[LOAD_VPTR]] : !cir.vptr -> !cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>
// CIR-AFTER: %[[GET_TYPEINFO:.*]] = cir.ptr_stride %[[VPTR_CAST]], %[[NEG_ONE]] : (!cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>, !s64i) -> !cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>
//
// CIR-NEXT: %[[LOAD_TYPEINFO:.*]] = cir.load align(8) %[[GET_TYPEINFO]] : !cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>, !cir.ptr<!rec_std3A3Atype_info>
// CIR-NEXT: cir.store align(8) %[[LOAD_TYPEINFO]], %[[GET_GLOB_ATI]] : !cir.ptr<!rec_std3A3Atype_info>, !cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>
// CIR-AFTER: cir.return
// CIR-NEXT:}
// CIR: cir.global "private" constant external @_ZN5Test11aE : !cir.ptr<!rec_Test13A3AA>
// LLVM: @_ZN5Test14a_tiE ={{.*}} global
const std::type_info &a_ti = typeid(a);

// CIR: cir.global constant external @_ZN5Test18A10_c_tiE = #cir.global_view<@_ZTIA10_c> : !cir.ptr<!rec_std3A3Atype_info>
// LLVM: @_ZN5Test18A10_c_tiE ={{.*}} constant ptr @_ZTIA10_c, align 8
const std::type_info &A10_c_ti = typeid(char const[10]);

// CIR: cir.func private dso_local @__cxa_bad_typeid() attributes {noreturn}

// CIR-LABEL: cir.func{{.*}} @_ZN5Test11fEPv
// CIR-SAME:  personality(@__gxx_personality_v0)
// LLVM-LABEL: define{{.*}} ptr @_ZN5Test11fEPv
// LLVM-SAME:  personality ptr @__gxx_personality_v0
const char *f(void *arg) {
  // CIR: %[[ARG:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["arg", init]
  try {
    // CIR: %[[ARG_VALUE:.*]] = cir.load{{.*}}%[[ARG]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
    // CIR-NEXT: %[[ARG_CAST:.*]] = cir.cast bitcast %[[ARG_VALUE]] : !cir.ptr<!void> -> !cir.ptr<!rec_Test13A3AA>
    // CIR-NEXT: %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_Test13A3AA>
    // CIR-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[ARG_CAST]], %[[NULL]])
    // CIR-NEXT: cir.if %[[CMP]] {
    // CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
    // CIR-NEXT: cir.unreachable
    // CIR-NEXT: }
    //
    // CIR: %[[GETVPTR:.*]] = cir.vtable.get_vptr %[[ARG_CAST]]
    // CIR-NEXT: %[[LOAD_VPTR:.*]] = cir.load{{.*}} %[[GETVPTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
    //
    // CIR-BEFORE: %[[GET_TYPEINFO:.*]] = cir.vtable.get_type_info %[[LOAD_VPTR]] : !cir.vptr -> !cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>
    //
    // CIR-AFTER: %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
    // CIR-AFTER: %[[VPTR_CAST:.*]] = cir.cast bitcast %[[LOAD_VPTR]] : !cir.vptr -> !cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>
    // CIR-AFTER: %[[GET_TYPEINFO:.*]] = cir.ptr_stride %[[VPTR_CAST]], %[[NEG_ONE]] : (!cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>, !s64i) -> !cir.ptr<!cir.ptr<!rec_std3A3Atype_info>>
    //
    // CIR-NEXT: %[[LOAD_TYPEINFO:.*]] = cir.load{{.*}}%[[GET_TYPEINFO]]
    // CIR-NEXT: cir.call @_ZNKSt9type_info4nameEv(%[[LOAD_TYPEINFO]])

    // LLVM: br i1
    // LLVM: invoke void @__cxa_bad_typeid()
    return typeid(*static_cast<A *>(arg)).name();
  } catch (...) {
    // LLVM:      landingpad { ptr, i32 }
    // LLVM-NEXT:   catch ptr null
  }

  return 0;
}

}
