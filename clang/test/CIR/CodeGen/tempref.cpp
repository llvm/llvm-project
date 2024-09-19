// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: cir-translate %t.cir -cir-to-llvmir -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct A { ~A(); };
A &&a = dynamic_cast<A&&>(A{});

//      CHECK: cir.func private  @_ZN1AD1Ev(!cir.ptr<!ty_A>) extra(#fn_attr)
// CHECK-NEXT: cir.global  external @a = #cir.ptr<null> : !cir.ptr<!ty_A> {alignment = 8 : i64, ast = #cir.var.decl.ast}
// CHECK-NEXT: cir.func internal private  @__cxx_global_var_init() {
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %[[SEVEN:[0-9]+]] = cir.get_global @a : !cir.ptr<!cir.ptr<!ty_A>>
// CHECK-NEXT:     %[[EIGHT:[0-9]+]] = cir.get_global @_ZGR1a_ : !cir.ptr<!ty_A>
// CHECK-NEXT:     cir.store %[[EIGHT]], %[[SEVEN]] : !cir.ptr<!ty_A>, !cir.ptr<!cir.ptr<!ty_A>>
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
// CHECK-NEXT: cir.func private  @_GLOBAL__sub_I_tempref.cpp() {
// CHECK-NEXT:   cir.call @__cxx_global_var_init() : () -> ()
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      LLVM: @_ZGR1a_ = internal global %struct.A undef
// LLVM-DAG: @a = global ptr null, align 8
// LLVM-DAG: @llvm.global_ctors = appending constant [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65536, ptr @__cxx_global_var_init, ptr null }]

// LLVM-DAG: declare {{.*}} void @_ZN1AD1Ev(ptr)

// LLVM-DAG: define internal void @__cxx_global_var_init()
// LLVM-DAG:   br label %[[L1:[0-9]+]]
// LLVM-DAG: [[L1]]:
// LLVM-DAG:   store ptr @_ZGR1a_, ptr @a, align 8
// LLVM-DAG:   br label %[[L2:[0-9]+]]
// LLVM-DAG: [[L2]]:
// LLVM-DAG:   ret void
// LLVM-DAG: }

// LLVM-DAG: define void @_GLOBAL__sub_I_tempref.cpp()
// LLVM-DAG:   call void @__cxx_global_var_init()
// LLVM-DAG:   ret void
// LLVM-DAG: }
