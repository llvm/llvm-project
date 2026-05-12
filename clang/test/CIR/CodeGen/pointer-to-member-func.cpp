// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefixes=CIR-AFTER,CIR-AFTER-X86 --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefixes=LLVM,LLVM-X86 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=OGCG,OGCG-X86 --input-file=%t.ll %s

// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t-arm.cir 2> %t-arm-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-arm-before.cir %s
// RUN: FileCheck --check-prefixes=CIR-AFTER,CIR-AFTER-ARM --input-file=%t-arm.cir %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-arm-cir.ll
// RUN: FileCheck --input-file=%t-arm-cir.ll --check-prefixes=LLVM,LLVM-ARM %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t-arm.ll
// RUN: FileCheck --check-prefixes=OGCG,OGCG-ARM --input-file=%t-arm.ll %s

// FIXME: Some of the differences between LLVM (via CIR) and OGCG below are
//        due to calling convention lowering being missing in the CIR path.

struct Foo {
  void m1(int);
  virtual void m2(int);
  virtual void m3(int);
};

// Global pointer to non-virtual method
void (Foo::*m1_ptr)(int) = &Foo::m1;

// CIR-BEFORE: cir.global external @m1_ptr = #cir.method<@_ZN3Foo2m1Ei> : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-AFTER-DAG:     cir.global "private" constant cir_private @[[NONVIRT_RET:.*]] = #cir.const_record<{#cir.global_view<@_ZN3Foo2m1Ei> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct
// CIR-AFTER-X86-DAG: cir.global "private" constant cir_private @[[VIRT_RET:.*]] = #cir.const_record<{#cir.int<9> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct
// CIR-AFTER-ARM-DAG: cir.global "private" constant cir_private @[[VIRT_RET:.*]] = #cir.const_record<{#cir.int<8> : !s64i, #cir.int<1> : !s64i}> : !rec_anon_struct
// CIR-AFTER-DAG:     cir.global "private" constant cir_private @[[NULL_RET:.*]] = #cir.const_record<{#cir.int<0> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct
// CIR-AFTER:         cir.global external @m1_ptr = #cir.const_record<{#cir.global_view<@_ZN3Foo2m1Ei> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct
// LLVM-DAG:     @m1_ptr = global { i64, i64 } { i64 ptrtoint (ptr @_ZN3Foo2m1Ei to i64), i64 0 }
// LLVM-DAG:     @[[NONVIRT_RET:.*]] = private constant { i64, i64 } { i64 ptrtoint (ptr @_ZN3Foo2m1Ei to i64), i64 0 }
// LLVM-X86-DAG: @[[VIRT_RET:.*]] = private constant { i64, i64 } { i64 9, i64 0 }
// LLVM-ARM-DAG: @[[VIRT_RET:.*]] = private constant { i64, i64 } { i64 8, i64 1 }
// LLVM-DAG:     @[[NULL_RET:.*]] = private constant { i64, i64 } zeroinitializer
// OGCG: @m1_ptr = global { i64, i64 } { i64 ptrtoint (ptr @_ZN3Foo2m1Ei to i64), i64 0 }

// Global pointer to virtual method
void (Foo::*m2_ptr)(int) = &Foo::m2;

// CIR-BEFORE: cir.global external @m2_ptr = #cir.method<vtable_offset = 0> : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-AFTER-X86: cir.global external @m2_ptr = #cir.const_record<{#cir.int<1> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct
// CIR-AFTER-ARM: cir.global external @m2_ptr = #cir.const_record<{#cir.int<0> : !s64i, #cir.int<1> : !s64i}> : !rec_anon_struct
// LLVM-X86-DAG: @m2_ptr = global { i64, i64 } { i64 1, i64 0 }
// LLVM-ARM-DAG: @m2_ptr = global { i64, i64 } { i64 0, i64 1 }
// OGCG-X86: @m2_ptr = global { i64, i64 } { i64 1, i64 0 }
// OGCG-ARM: @m2_ptr = global { i64, i64 } { i64 0, i64 1 }

// Self-referencing PMF causes a null method.
long (Foo::*pmf1)(int) = pmf1;
// CIR-BEFORE: @pmf1 = ctor : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i) -> !s64i> in !rec_Foo> {
// CIR-AFTER: cir.global external @pmf1 = #cir.const_record<{#cir.int<0> : !s64i, #cir.int<0> : !s64i}> 
// LLVM: @pmf1 = global { i64, i64 } zeroinitializer, align 8 
// OGCG: @pmf1 = global { i64, i64 } zeroinitializer, align 8 

auto make_non_virtual() -> void (Foo::*)(int) {
  return &Foo::m1;
}

// CIR-BEFORE: cir.func {{.*}} @_Z16make_non_virtualv() -> !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   %[[RETVAL:.*]] = cir.alloca !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>, !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>>, ["__retval"]
// CIR-BEFORE:   %[[METHOD_PTR:.*]] = cir.const #cir.method<@_ZN3Foo2m1Ei> : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   cir.store %[[METHOD_PTR]], %[[RETVAL]]
// CIR-BEFORE:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR-BEFORE:   cir.return %[[RET]] : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>

// CIR-AFTER: cir.func {{.*}} @_Z16make_non_virtualv() -> !rec_anon_struct attributes {{{.*}}nothrow} {
// CIR-AFTER:   %[[RETVAL:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["__retval"]
// CIR-AFTER:   %[[METHOD_PTR:.*]] = cir.get_global @[[NONVIRT_RET]] : !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   cir.copy %[[METHOD_PTR]] to %[[RETVAL]] : !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR-AFTER:   cir.return %[[RET]] : !rec_anon_struct

// LLVM: define {{.*}} { i64, i64 } @_Z16make_non_virtualv()
// LLVM:   %[[RETVAL:.*]] = alloca { i64, i64 }
// LLVM:   call void @llvm.memcpy{{.*}}(ptr %[[RETVAL]], ptr @[[NONVIRT_RET]]
// LLVM:   %[[RET:.*]] = load { i64, i64 }, ptr %[[RETVAL]]
// LLVM:   ret { i64, i64 } %[[RET]]

// OGCG-X86: define {{.*}} { i64, i64 } @_Z16make_non_virtualv()
// OGCG-X86:   ret { i64, i64 } { i64 ptrtoint (ptr @_ZN3Foo2m1Ei to i64), i64 0 }

// OGCG-ARM: define {{.*}} [2 x i64] @_Z16make_non_virtualv()
// OGCG-ARM:   %[[RETVAL:.*]] = alloca { i64, i64 }
// OGCG-ARM:   store { i64, i64 } { i64 ptrtoint (ptr @_ZN3Foo2m1Ei to i64), i64 0 }, ptr %[[RETVAL]]
// OGCG-ARM:   %[[RET:.*]] = load [2 x i64], ptr %[[RETVAL]]
// OGCG-ARM:   ret [2 x i64] %[[RET]]

auto make_virtual() -> void (Foo::*)(int) {
  return &Foo::m3;
}

// CIR-BEFORE: cir.func {{.*}} @_Z12make_virtualv() -> !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   %[[RETVAL:.*]] = cir.alloca !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>, !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>>, ["__retval"]
// CIR-BEFORE:   %[[METHOD_PTR:.*]] = cir.const #cir.method<vtable_offset = 8> : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   cir.store %[[METHOD_PTR]], %[[RETVAL]]
// CIR-BEFORE:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>>, !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   cir.return %[[RET]] : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>

// CIR-AFTER: cir.func {{.*}} @_Z12make_virtualv() -> !rec_anon_struct
// CIR-AFTER:   %[[RETVAL:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["__retval"]
// CIR-AFTER:   %[[METHOD_PTR:.*]] = cir.get_global @[[VIRT_RET]] : !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   cir.copy %[[METHOD_PTR]] to %[[RETVAL]] : !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR-AFTER:   cir.return %[[RET]] : !rec_anon_struct

// LLVM: define {{.*}} @_Z12make_virtualv()
// LLVM:   %[[RETVAL:.*]] = alloca { i64, i64 }
// LLVM:   call void @llvm.memcpy{{.*}}(ptr %[[RETVAL]], ptr @[[VIRT_RET]]
// LLVM:   %[[RET:.*]] = load { i64, i64 }, ptr %[[RETVAL]]
// LLVM:   ret { i64, i64 } %[[RET]]

// OGCG:     define {{.*}} @_Z12make_virtualv()
// OGCG-X86:   ret { i64, i64 } { i64 9, i64 0 }
// OGCG-ARM:   %[[RETVAL:.*]] = alloca { i64, i64 }
// OGCG-ARM:   store { i64, i64 } { i64 8, i64 1 }, ptr %[[RETVAL]]
// OGCG-ARM:   %[[RET:.*]] = load [2 x i64], ptr %[[RETVAL]]
// OGCG-ARM:   ret [2 x i64] %[[RET]]

auto make_null() -> void (Foo::*)(int) {
  return nullptr;
}

// CIR-BEFORE: cir.func {{.*}} @_Z9make_nullv() -> !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   %[[RETVAL:.*]] = cir.alloca !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>, !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>>, ["__retval"]
// CIR-BEFORE:   %[[METHOD_PTR:.*]] = cir.const #cir.method<null> : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   cir.store %[[METHOD_PTR]], %[[RETVAL]]
// CIR-BEFORE:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR-BEFORE:   cir.return %[[RET]] : !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>

// CIR-AFTER: cir.func {{.*}} @_Z9make_nullv() -> !rec_anon_struct
// CIR-AFTER:   %[[RETVAL:.*]] = cir.alloca !rec_anon_struct, !cir.ptr<!rec_anon_struct>, ["__retval"]
// CIR-AFTER:   %[[METHOD_PTR:.*]] = cir.get_global @[[NULL_RET]] : !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   cir.copy %[[METHOD_PTR]] to %[[RETVAL]] : !cir.ptr<!rec_anon_struct>
// CIR-AFTER:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR-AFTER:   cir.return %[[RET]] : !rec_anon_struct

// LLVM: define {{.*}} @_Z9make_nullv()
// LLVM:   %[[RETVAL:.*]] = alloca { i64, i64 }
// LLVM:   call void @llvm.memcpy{{.*}}(ptr %[[RETVAL]], ptr @[[NULL_RET]]
// LLVM:   %[[RET:.*]] = load { i64, i64 }, ptr %[[RETVAL]]
// LLVM:   ret { i64, i64 } %[[RET]]

// OGCG:     define {{.*}} @_Z9make_nullv()
// OGCG-X86:   ret { i64, i64 } zeroinitializer
// OGCG-ARM:   %[[RETVAL:.*]] = alloca { i64, i64 }
// OGCG-ARM:   store { i64, i64 } zeroinitializer, ptr %[[RETVAL]]
// OGCG-ARM:   %[[RET:.*]] = load [2 x i64], ptr %[[RETVAL]]
// OGCG-ARM:   ret [2 x i64] %[[RET]]

void call(Foo *obj, void (Foo::*func)(int), int arg) {
  (obj->*func)(arg);
}

// CIR-BEFORE: cir.func {{.*}} @_Z4callP3FooMS_FviEi
// CIR-BEFORE:   %[[OBJ:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!rec_Foo>>, !cir.ptr<!rec_Foo>
// CIR-BEFORE:   %[[FUNC:.*]] = cir.load{{.*}} : !cir.ptr<!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>>, !cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>
// CIR-BEFORE:   %[[CALLEE:.*]], %[[THIS:.*]] = cir.get_method %[[FUNC]], %[[OBJ]] : (!cir.method<!cir.func<(!cir.ptr<!rec_Foo>, !s32i)> in !rec_Foo>, !cir.ptr<!rec_Foo>) -> (!cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>, !cir.ptr<!void>)
// CIR-BEFORE:   %[[ARG:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR-BEFORE:   cir.call %[[CALLEE]](%[[THIS]], %[[ARG]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>, !cir.ptr<!void> {{.*}}, !s32i {{.*}}) -> ()

// CIR-AFTER:    cir.func {{.*}} @_Z4callP3FooMS_FviEi
// CIR-AFTER:      %[[OBJ:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!rec_Foo>>, !cir.ptr<!rec_Foo>
// CIR-AFTER:      %[[FUNC:.*]] = cir.load{{.*}} : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR-AFTER:      %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR-AFTER:      %[[ADJ:.*]] = cir.extract_member %[[FUNC]][1] : !rec_anon_struct -> !s64i
// CIR-AFTER-ARM:  %[[ADJ_SHIFT:.*]] = cir.shift(right, %[[ADJ]] : !s64i, %[[ONE:.*]] : !s64i) -> !s64i
// CIR-AFTER:      %[[THIS:.*]] = cir.cast bitcast %[[OBJ]] : !cir.ptr<!rec_Foo> -> !cir.ptr<!void>
// CIR-AFTER-X86:  %[[ADJUSTED_THIS:.*]] = cir.ptr_stride %[[THIS]], %[[ADJ]] : (!cir.ptr<!void>, !s64i) -> !cir.ptr<!void>
// CIR-AFTER-ARM:  %[[ADJUSTED_THIS:.*]] = cir.ptr_stride %[[THIS]], %[[ADJ_SHIFT]] : (!cir.ptr<!void>, !s64i) -> !cir.ptr<!void>
// CIR-AFTER:      %[[METHOD_PTR:.*]] = cir.extract_member %[[FUNC]][0] : !rec_anon_struct -> !s64i
// CIR-AFTER-X86:  %[[VIRT_BIT_TEST:.*]] = cir.and %[[METHOD_PTR]], %[[ONE]] : !s64i
// CIR-AFTER-ARM:  %[[VIRT_BIT_TEST:.*]] = cir.and %[[ADJ]], %[[ONE]] : !s64i
// CIR-AFTER:      %[[IS_VIRTUAL:.*]] = cir.cmp eq %[[VIRT_BIT_TEST]], %[[ONE]] : !s64i
// CIR-AFTER:      %[[CALLEE:.*]] = cir.ternary(%[[IS_VIRTUAL]], true {
// CIR-AFTER:        %[[VTABLE_PTR:.*]] = cir.cast bitcast %[[ADJUSTED_THIS]] : !cir.ptr<!void> -> !cir.ptr<!cir.ptr<!s8i>>
// CIR-AFTER:        %[[VTABLE:.*]] = cir.load %[[VTABLE_PTR]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CIR-AFTER-X86:    %[[OFFSET:.*]] = cir.sub %[[METHOD_PTR]], %[[ONE]] : !s64i
// CIR-AFTER-X86:    %[[VTABLE_SLOT:.*]] = cir.ptr_stride %[[VTABLE]], %[[OFFSET]] : (!cir.ptr<!s8i>, !s64i) -> !cir.ptr<!s8i>
// CIR-AFTER-ARM:    %[[VTABLE_SLOT:.*]] = cir.ptr_stride %[[VTABLE]], %[[METHOD_PTR]] : (!cir.ptr<!s8i>, !s64i) -> !cir.ptr<!s8i>
// CIR-AFTER:        %[[VIRTUAL_FN_PTR:.*]] = cir.cast bitcast %[[VTABLE_SLOT]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>>
// CIR-AFTER:        %[[VIRTUAL_FN_PTR_LOAD:.*]] = cir.load %[[VIRTUAL_FN_PTR]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>>, !cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>
// CIR-AFTER:        cir.yield %[[VIRTUAL_FN_PTR_LOAD]] : !cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>
// CIR-AFTER:      }, false {
// CIR-AFTER:        %[[CALLEE_PTR:.*]] = cir.cast int_to_ptr %[[METHOD_PTR]] : !s64i -> !cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>
// CIR-AFTER:        cir.yield %[[CALLEE_PTR]] : !cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>
// CIR-AFTER:      }) : (!cir.bool) -> !cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>
// CIR-AFTER:      %[[ARG:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR-AFTER:      cir.call %[[CALLEE]](%[[ADJUSTED_THIS]], %[[ARG]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>, !cir.ptr<!rec_Foo>, !s32i)>>, !cir.ptr<!void> {{.*}}, !s32i {{.*}}) -> ()

// LLVM:     define {{.*}} @_Z4callP3FooMS_FviEi
// LLVM:       %[[OBJ:.*]] = load ptr, ptr %{{.*}}
// LLVM:       %[[MEMFN_PTR:.*]] = load { i64, i64 }, ptr %{{.*}}
// LLVM:       %[[THIS_ADJ:.*]] = extractvalue { i64, i64 } %[[MEMFN_PTR]], 1
// LLVM-X86:   %[[ADJUSTED_THIS:.*]] = getelementptr i8, ptr %[[OBJ]], i64 %[[THIS_ADJ]]
// LLVM-ARM:   %[[THIS_ADJ_SHIFT:.*]] = ashr i64 %[[THIS_ADJ]], 1
// LLVM-ARM:   %[[ADJUSTED_THIS:.*]] = getelementptr i8, ptr %[[OBJ]], i64 %[[THIS_ADJ_SHIFT]]
// LLVM:       %[[PTR_FIELD:.*]] = extractvalue { i64, i64 } %[[MEMFN_PTR]], 0
// LLVM-ARM:   %[[VIRT_BIT:.*]] = and i64 %[[THIS_ADJ]], 1
// LLVM-X86:   %[[VIRT_BIT:.*]] = and i64 %[[PTR_FIELD]], 1
// LLVM:       %[[IS_VIRTUAL:.*]] = icmp eq i64 %[[VIRT_BIT]], 1
// LLVM:       br i1 %[[IS_VIRTUAL]], label %[[HANDLE_VIRTUAL:.*]], label %[[HANDLE_NON_VIRTUAL:.*]]
// LLVM:     [[HANDLE_VIRTUAL]]:
// LLVM:       %[[VTABLE:.*]] = load ptr, ptr %[[ADJUSTED_THIS]]
// LLVM-X86:   %[[OFFSET:.*]] = sub i64 %[[PTR_FIELD]], 1
// LLVM-X86:   %[[VTABLE_SLOT:.*]] = getelementptr i8, ptr %[[VTABLE]], i64 %[[OFFSET]]
// LLVM-ARM:   %[[VTABLE_SLOT:.*]] = getelementptr i8, ptr %[[VTABLE]], i64 %[[PTR_FIELD]]
// LLVM:       %[[VIRTUAL_FN_PTR:.*]] = load ptr, ptr %[[VTABLE_SLOT]]
// LLVM:       br label %[[CONTINUE:.*]]
// LLVM:     [[HANDLE_NON_VIRTUAL]]:
// LLVM:       %[[FUNC_PTR:.*]] = inttoptr i64 %[[PTR_FIELD]] to ptr
// LLVM:       br label %[[CONTINUE]]
// LLVM:     [[CONTINUE]]:
// LLVM:       %[[CALLEE_PTR:.*]] = phi ptr [ %[[FUNC_PTR]], %[[HANDLE_NON_VIRTUAL]] ], [ %[[VIRTUAL_FN_PTR]], %[[HANDLE_VIRTUAL]] ]
// LLVM:       %[[ARG:.*]] = load i32, ptr %{{.+}}
// LLVM:       call void %[[CALLEE_PTR]](ptr {{.*}} %[[ADJUSTED_THIS]], i32 {{.*}} %[[ARG]])
// LLVM:     }

// OGCG:     define {{.*}} @_Z4callP3FooMS_FviEi
// OGCG:       %[[OBJ:.*]] = load ptr, ptr %{{.*}}
// OGCG:       %[[MEMFN_PTR:.*]] = load { i64, i64 }, ptr %{{.*}}
// OGCG:       %[[THIS_ADJ:.*]] = extractvalue { i64, i64 } %[[MEMFN_PTR]], 1
// OGCG-X86:   %[[ADJUSTED_THIS:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i64 %[[THIS_ADJ]]
// OGCG-ARM:   %[[THIS_ADJ_SHIFT:.*]] = ashr i64 %[[THIS_ADJ]], 1
// OGCG-ARM:   %[[ADJUSTED_THIS:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i64 %[[THIS_ADJ_SHIFT]]
// OGCG:       %[[PTR_FIELD:.*]] = extractvalue { i64, i64 } %[[MEMFN_PTR]], 0
// OGCG-X86:   %[[VIRT_BIT:.*]] = and i64 %[[PTR_FIELD]], 1
// OGCG-ARM:   %[[VIRT_BIT:.*]] = and i64 %[[THIS_ADJ]], 1
// OGCG:       %[[IS_VIRTUAL:.*]] = icmp ne i64 %[[VIRT_BIT]], 0
// OGCG:       br i1 %[[IS_VIRTUAL]], label %[[HANDLE_VIRTUAL:.*]], label %[[HANDLE_NON_VIRTUAL:.*]]
// OGCG:     [[HANDLE_VIRTUAL]]:
// OGCG:       %[[VTABLE:.*]] = load ptr, ptr %[[ADJUSTED_THIS]]
// OGCG-X86:   %[[OFFSET:.*]] = sub i64 %[[PTR_FIELD]], 1
// OGCG-X86:   %[[VTABLE_SLOT:.*]] = getelementptr i8, ptr %[[VTABLE]], i64 %[[OFFSET]]
// OGCG-ARM:   %[[VTABLE_SLOT:.*]] = getelementptr i8, ptr %[[VTABLE]], i64 %[[PTR_FIELD]]
// OGCG:       %[[VIRTUAL_FN_PTR:.*]] = load ptr, ptr %[[VTABLE_SLOT]]
// OGCG:       br label %[[CONTINUE:.*]]
// OGCG:     [[HANDLE_NON_VIRTUAL]]:
// OGCG:       %[[FUNC_PTR:.*]] = inttoptr i64 %[[PTR_FIELD]] to ptr
// OGCG:       br label %[[CONTINUE]]
// OGCG:     [[CONTINUE]]:
// OGCG:       %[[CALLEE_PTR:.*]] = phi ptr [ %[[VIRTUAL_FN_PTR]], %[[HANDLE_VIRTUAL]] ], [ %[[FUNC_PTR]], %[[HANDLE_NON_VIRTUAL]] ]
// OGCG:       %[[ARG:.*]] = load i32, ptr %{{.+}}
// OGCG:       call void %[[CALLEE_PTR]](ptr {{.*}} %[[ADJUSTED_THIS]], i32 {{.*}} %[[ARG]])
// OGCG:     }
