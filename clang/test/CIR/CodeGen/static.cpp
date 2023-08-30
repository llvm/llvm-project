// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER
// RUN: cir-opt %t.cir -o - | FileCheck %s -check-prefix=AFTER
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

class Init {

public:
  Init(bool a) ;
  ~Init();
private:
  static bool _S_synced_with_stdio;
};


static Init __ioinit(true);
static Init __ioinit2(false);

// BEFORE:      module {{.*}} {
// BEFORE-NEXT:   cir.func private @_ZN4InitC1Eb(!cir.ptr<!ty_22Init22>, !cir.bool)
// BEFORE-NEXT:   cir.func private @_ZN4InitD1Ev(!cir.ptr<!ty_22Init22>)
// BEFORE-NEXT:   cir.global "private" internal @_ZL8__ioinit = ctor : !ty_22Init22 {
// BEFORE-NEXT:     %0 = cir.get_global @_ZL8__ioinit : cir.ptr <!ty_22Init22>
// BEFORE-NEXT:     %1 = cir.const(#true) : !cir.bool
// BEFORE-NEXT:     cir.call @_ZN4InitC1Eb(%0, %1) : (!cir.ptr<!ty_22Init22>, !cir.bool) -> ()
// BEFORE-NEXT:   } dtor {
// BEFORE-NEXT:      %0 = cir.get_global @_ZL8__ioinit : cir.ptr <!ty_22Init22>
// BEFORE-NEXT:      cir.call @_ZN4InitD1Ev(%0) : (!cir.ptr<!ty_22Init22>) -> ()
// BEFORE-NEXT:   } {ast = #cir.vardecl.ast}
// BEFORE:        cir.global "private" internal @_ZL9__ioinit2 = ctor : !ty_22Init22 {
// BEFORE-NEXT:     %0 = cir.get_global @_ZL9__ioinit2 : cir.ptr <!ty_22Init22>
// BEFORE-NEXT:     %1 = cir.const(#false) : !cir.bool
// BEFORE-NEXT:     cir.call @_ZN4InitC1Eb(%0, %1) : (!cir.ptr<!ty_22Init22>, !cir.bool) -> ()
// BEFORE-NEXT:   } dtor  {
// BEFORE-NEXT:     %0 = cir.get_global @_ZL9__ioinit2 : cir.ptr <!ty_22Init22>
// BEFORE-NEXT:     cir.call @_ZN4InitD1Ev(%0) : (!cir.ptr<!ty_22Init22>) -> ()
// BEFORE-NEXT:   } {ast = #cir.vardecl.ast}
// BEFORE-NEXT: }


// AFTER:      module {{.*}} attributes {{.*}}cir.globalCtors = [#cir.globalCtor<"__cxx_global_var_init">, #cir.globalCtor<"__cxx_global_var_init.1">]
// AFTER-NEXT:   cir.global "private" external @__dso_handle : i8
// AFTER-NEXT:   cir.func private @__cxa_atexit(!cir.ptr<!cir.func<!void (!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>)
// AFTER-NEXT:   cir.func private @_ZN4InitC1Eb(!cir.ptr<!ty_22Init22>, !cir.bool)
// AFTER-NEXT:   cir.func private @_ZN4InitD1Ev(!cir.ptr<!ty_22Init22>)
// AFTER-NEXT:   cir.global "private" internal @_ZL8__ioinit =  #cir.zero : !ty_22Init22 {ast = #cir.vardecl.ast}
// AFTER-NEXT:   cir.func internal private @__cxx_global_var_init()
// AFTER-NEXT:     %0 = cir.get_global @_ZL8__ioinit : cir.ptr <!ty_22Init22>
// AFTER-NEXT:     %1 = cir.const(#true) : !cir.bool
// AFTER-NEXT:     cir.call @_ZN4InitC1Eb(%0, %1) : (!cir.ptr<!ty_22Init22>, !cir.bool) -> ()
// AFTER-NEXT:     %2 = cir.get_global @_ZL8__ioinit : cir.ptr <!ty_22Init22>
// AFTER-NEXT:     %3 = cir.get_global @_ZN4InitD1Ev : cir.ptr <!cir.func<!void (!cir.ptr<!ty_22Init22>)>>
// AFTER-NEXT:     %4 = cir.cast(bitcast, %3 : !cir.ptr<!cir.func<!void (!cir.ptr<!ty_22Init22>)>>), !cir.ptr<!cir.func<!void (!cir.ptr<!void>)>>
// AFTER-NEXT:     %5 = cir.cast(bitcast, %2 : !cir.ptr<!ty_22Init22>), !cir.ptr<!void>
// AFTER-NEXT:     %6 = cir.get_global @__dso_handle : cir.ptr <i8>
// AFTER-NEXT:     cir.call @__cxa_atexit(%4, %5, %6) : (!cir.ptr<!cir.func<!void (!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// AFTER-NEXT:     cir.return
// AFTER:        cir.global "private" internal @_ZL9__ioinit2 =  #cir.zero : !ty_22Init22 {ast = #cir.vardecl.ast}
// AFTER-NEXT:   cir.func internal private @__cxx_global_var_init.1()
// AFTER-NEXT:     %0 = cir.get_global @_ZL9__ioinit2 : cir.ptr <!ty_22Init22>
// AFTER-NEXT:     %1 = cir.const(#false) : !cir.bool
// AFTER-NEXT:     cir.call @_ZN4InitC1Eb(%0, %1) : (!cir.ptr<!ty_22Init22>, !cir.bool) -> ()
// AFTER-NEXT:     %2 = cir.get_global @_ZL9__ioinit2 : cir.ptr <!ty_22Init22>
// AFTER-NEXT:     %3 = cir.get_global @_ZN4InitD1Ev : cir.ptr <!cir.func<!void (!cir.ptr<!ty_22Init22>)>>
// AFTER-NEXT:     %4 = cir.cast(bitcast, %3 : !cir.ptr<!cir.func<!void (!cir.ptr<!ty_22Init22>)>>), !cir.ptr<!cir.func<!void (!cir.ptr<!void>)>>
// AFTER-NEXT:     %5 = cir.cast(bitcast, %2 : !cir.ptr<!ty_22Init22>), !cir.ptr<!void>
// AFTER-NEXT:     %6 = cir.get_global @__dso_handle : cir.ptr <i8>
// AFTER-NEXT:     cir.call @__cxa_atexit(%4, %5, %6) : (!cir.ptr<!cir.func<!void (!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
// AFTER-NEXT:     cir.return
// AFTER:        cir.func private @_GLOBAL__sub_I_static.cpp()
// AFTER-NEXT:     cir.call @__cxx_global_var_init() : () -> ()
// AFTER-NEXT:     cir.call @__cxx_global_var_init.1() : () -> ()
// AFTER-NEXT:     cir.return

// LLVM:      @__dso_handle = external global i8
// LLVM:      @_ZL8__ioinit = internal global %class.Init zeroinitializer
// LLVM:      @_ZL9__ioinit2 = internal global %class.Init zeroinitializer
// LLVM:      @llvm.global_ctors = appending constant [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65536, ptr @__cxx_global_var_init, ptr null }, { i32, ptr, ptr } { i32 65536, ptr @__cxx_global_var_init.1, ptr null }]
// LLVM:      define internal void @__cxx_global_var_init()
// LLVM-NEXT:   call void @_ZN4InitC1Eb(ptr @_ZL8__ioinit, i8 1)
// LLVM-NEXT:   call void @__cxa_atexit(ptr @_ZN4InitD1Ev, ptr @_ZL8__ioinit, ptr @__dso_handle)
// LLVM-NEXT:   ret void
// LLVM:      define internal void @__cxx_global_var_init.1()
// LLVM-NEXT:   call void @_ZN4InitC1Eb(ptr @_ZL9__ioinit2, i8 0)
// LLVM-NEXT:   call void @__cxa_atexit(ptr @_ZN4InitD1Ev, ptr @_ZL9__ioinit2, ptr @__dso_handle)
// LLVM-NEXT:   ret void
// LLVM:      define void @_GLOBAL__sub_I_static.cpp()
// LLVM-NEXT:  call void @__cxx_global_var_init()
// LLVM-NEXT:  call void @__cxx_global_var_init.1()
// LLVM-NEXT:  ret void
