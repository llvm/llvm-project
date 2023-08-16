// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER
// RUN: cir-opt %t.cir -o - | FileCheck %s -check-prefix=AFTER
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

class Init {

public:
  Init(bool a) ;

private:
  static bool _S_synced_with_stdio;
};


static Init __ioinit(true);
static Init __ioinit2(false);

// BEFORE:      module {{.*}} {
// BEFORE-NEXT:   cir.func private @_ZN4InitC1Eb(!cir.ptr<!ty_22class2EInit22>, !cir.bool)
// BEFORE-NEXT:   cir.global "private" internal @_ZL8__ioinit = ctor : !ty_22class2EInit22 {
// BEFORE-NEXT:     %0 = cir.get_global @_ZL8__ioinit : cir.ptr <!ty_22class2EInit22>
// BEFORE-NEXT:     %1 = cir.const(#true) : !cir.bool
// BEFORE-NEXT:     cir.call @_ZN4InitC1Eb(%0, %1) : (!cir.ptr<!ty_22class2EInit22>, !cir.bool) -> ()
// BEFORE-NEXT:   } {ast = #cir.vardecl.ast}
// BEFORE:        cir.global "private" internal @_ZL9__ioinit2 = ctor : !ty_22class2EInit22 {
// BEFORE-NEXT:     %0 = cir.get_global @_ZL9__ioinit2 : cir.ptr <!ty_22class2EInit22>
// BEFORE-NEXT:     %1 = cir.const(#false) : !cir.bool
// BEFORE-NEXT:     cir.call @_ZN4InitC1Eb(%0, %1) : (!cir.ptr<!ty_22class2EInit22>, !cir.bool) -> ()
// BEFORE-NEXT:   } {ast = #cir.vardecl.ast}
// BEFORE-NEXT: }


// AFTER:      module {{.*}} {
// AFTER-NEXT:   cir.func private @_ZN4InitC1Eb(!cir.ptr<!ty_22class2EInit22>, !cir.bool)
// AFTER-NEXT:   cir.global "private" internal @_ZL8__ioinit =  #cir.zero : !ty_22class2EInit22 {ast = #cir.vardecl.ast}
// AFTER-NEXT:   cir.func internal private @__cxx_global_var_init()
// AFTER-NEXT:     %0 = cir.get_global @_ZL8__ioinit : cir.ptr <!ty_22class2EInit22>
// AFTER-NEXT:     %1 = cir.const(#true) : !cir.bool
// AFTER-NEXT:     cir.call @_ZN4InitC1Eb(%0, %1) : (!cir.ptr<!ty_22class2EInit22>, !cir.bool) -> ()
// AFTER-NEXT:     cir.return
// AFTER:        cir.global "private" internal @_ZL9__ioinit2 =  #cir.zero : !ty_22class2EInit22 {ast = #cir.vardecl.ast}
// AFTER-NEXT:   cir.func internal private @__cxx_global_var_init.1()
// AFTER-NEXT:     %0 = cir.get_global @_ZL9__ioinit2 : cir.ptr <!ty_22class2EInit22>
// AFTER-NEXT:     %1 = cir.const(#false) : !cir.bool
// AFTER-NEXT:     cir.call @_ZN4InitC1Eb(%0, %1) : (!cir.ptr<!ty_22class2EInit22>, !cir.bool) -> ()
// AFTER-NEXT:     cir.return
// AFTER:        cir.func private @_GLOBAL__sub_I_static.cpp()
// AFTER-NEXT:     cir.call @__cxx_global_var_init() : () -> ()
// AFTER-NEXT:     cir.call @__cxx_global_var_init.1() : () -> ()
// AFTER-NEXT:     cir.return


// LLVM:      @_ZL8__ioinit = internal global %class.Init zeroinitializer
// LLVM:      @_ZL9__ioinit2 = internal global %class.Init zeroinitializer
// LLVM:      define internal void @__cxx_global_var_init()
// LLVM-NEXT:   call void @_ZN4InitC1Eb(ptr @_ZL8__ioinit, i8 1)
// LLVM-NEXT:   ret void
// LLVM:      define internal void @__cxx_global_var_init.1()
// LLVM-NEXT:   call void @_ZN4InitC1Eb(ptr @_ZL9__ioinit2, i8 0)
// LLVM-NEXT:   ret void
// LLVM:      define void @_GLOBAL__sub_I_static.cpp()
// LLVM-NEXT:  call void @__cxx_global_var_init()
// LLVM-NEXT:  call void @__cxx_global_var_init.1()
// LLVM-NEXT:  ret void
