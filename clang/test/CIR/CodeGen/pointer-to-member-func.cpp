// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

struct Foo {
  void m1(int);
  virtual void m2(int);
  virtual void m3(int);
};

auto make_non_virtual() -> void (Foo::*)(int) {
  return &Foo::m1;
}

// CHECK-LABEL: cir.func @_Z16make_non_virtualv() -> !cir.method<!cir.func<(!s32i)> in !ty_Foo>
//       CHECK:   %{{.+}} = cir.const #cir.method<@_ZN3Foo2m1Ei> : !cir.method<!cir.func<(!s32i)> in !ty_Foo>
//       CHECK: }

// LLVM-LABEL: @_Z16make_non_virtualv
//       LLVM:   store { i64, i64 } { i64 ptrtoint (ptr @_ZN3Foo2m1Ei to i64), i64 0 }, ptr %{{.+}}
//       LLVM: }

auto make_virtual() -> void (Foo::*)(int) {
  return &Foo::m3;
}

// CHECK-LABEL: cir.func @_Z12make_virtualv() -> !cir.method<!cir.func<(!s32i)> in !ty_Foo>
//       CHECK:   %{{.+}} = cir.const #cir.method<vtable_offset = 8> : !cir.method<!cir.func<(!s32i)> in !ty_Foo>
//       CHECK: }

// LLVM-LABEL: @_Z12make_virtualv
//       LLVM:   store { i64, i64 } { i64 9, i64 0 }, ptr %{{.+}}
//       LLVM: }

auto make_null() -> void (Foo::*)(int) {
  return nullptr;
}

// CHECK-LABEL: cir.func @_Z9make_nullv() -> !cir.method<!cir.func<(!s32i)> in !ty_Foo>
//       CHECK:   %{{.+}} = cir.const #cir.method<null> : !cir.method<!cir.func<(!s32i)> in !ty_Foo>
//       CHECK: }

// LLVM-LABEL: @_Z9make_nullv
//       LLVM:   store { i64, i64 } zeroinitializer, ptr %{{.+}}
//       LLVM: }

void call(Foo *obj, void (Foo::*func)(int), int arg) {
  (obj->*func)(arg);
}

// CHECK-LABEL: cir.func @_Z4callP3FooMS_FviEi
//       CHECK:   %[[CALLEE:.+]], %[[THIS:.+]] = cir.get_method %{{.+}}, %{{.+}} : (!cir.method<!cir.func<(!s32i)> in !ty_Foo>, !cir.ptr<!ty_Foo>) -> (!cir.ptr<!cir.func<(!cir.ptr<!void>, !s32i)>>, !cir.ptr<!void>)
//  CHECK-NEXT:   %[[#ARG:]] = cir.load %{{.+}} : !cir.ptr<!s32i>, !s32i
//  CHECK-NEXT:   cir.call %[[CALLEE]](%[[THIS]], %[[#ARG]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>, !s32i)>>, !cir.ptr<!void>, !s32i) -> ()
//       CHECK: }

// LLVM-LABEL: @_Z4callP3FooMS_FviEi
//      LLVM:    %[[#obj:]] = load ptr, ptr %{{.+}}
// LLVM-NEXT:    %[[#memfn_ptr:]] = load { i64, i64 }, ptr %{{.+}}
// LLVM-NEXT:    %[[#this_adj:]] = extractvalue { i64, i64 } %[[#memfn_ptr]], 1
// LLVM-NEXT:    %[[#adjusted_this:]] = getelementptr i8, ptr %[[#obj]], i64 %[[#this_adj]]
// LLVM-NEXT:    %[[#ptr_field:]] = extractvalue { i64, i64 } %[[#memfn_ptr]], 0
// LLVM-NEXT:    %[[#virt_bit:]] = and i64 %[[#ptr_field]], 1
// LLVM-NEXT:    %[[#is_virt:]] = icmp eq i64 %[[#virt_bit]], 1
// LLVM-NEXT:    br i1 %[[#is_virt]], label %[[#block_virt:]], label %[[#block_non_virt:]]
//      LLVM:  [[#block_virt]]:
// LLVM-NEXT:    %[[#vtable_ptr:]] = load ptr, ptr %[[#obj]]
// LLVM-NEXT:    %[[#vtable_offset:]] = sub i64 %[[#ptr_field]], 1
// LLVM-NEXT:    %[[#vfp_ptr:]] = getelementptr i8, ptr %[[#vtable_ptr]], i64 %[[#vtable_offset]]
// LLVM-NEXT:    %[[#vfp:]] = load ptr, ptr %[[#vfp_ptr]]
// LLVM-NEXT:    br label %[[#block_continue:]]
//      LLVM:  [[#block_non_virt]]:
// LLVM-NEXT:    %[[#func_ptr:]] = inttoptr i64 %[[#ptr_field]] to ptr
// LLVM-NEXT:    br label %[[#block_continue]]
//      LLVM:  [[#block_continue]]:
// LLVM-NEXT:    %[[#callee_ptr:]] = phi ptr [ %[[#func_ptr]], %[[#block_non_virt]] ], [ %[[#vfp]], %[[#block_virt]] ]
// LLVM-NEXT:    %[[#arg:]] = load i32, ptr %{{.+}}
// LLVM-NEXT:    call void %[[#callee_ptr]](ptr %[[#adjusted_this]], i32 %[[#arg]])
//      LLVM: }

bool cmp_eq(void (Foo::*lhs)(int), void (Foo::*rhs)(int)) {
  return lhs == rhs;
}

// CHECK-LABEL: @_Z6cmp_eqM3FooFviES1_
// CHECK: %{{.+}} = cir.cmp(eq, %{{.+}}, %{{.+}}) : !cir.method<!cir.func<(!s32i)> in !ty_Foo>, !cir.bool

// LLVM-LABEL: @_Z6cmp_eqM3FooFviES1_
//      LLVM: %[[#lhs:]] = load { i64, i64 }, ptr %{{.+}}
// LLVM-NEXT: %[[#rhs:]] = load { i64, i64 }, ptr %{{.+}}
// LLVM-NEXT: %[[#lhs_ptr:]] = extractvalue { i64, i64 } %[[#lhs]], 0
// LLVM-NEXT: %[[#rhs_ptr:]] = extractvalue { i64, i64 } %[[#rhs]], 0
// LLVM-NEXT: %[[#ptr_cmp:]] = icmp eq i64 %[[#lhs_ptr]], %[[#rhs_ptr]]
// LLVM-NEXT: %[[#ptr_null:]] = icmp eq i64 %[[#lhs_ptr]], 0
// LLVM-NEXT: %[[#lhs_adj:]] = extractvalue { i64, i64 } %[[#lhs]], 1
// LLVM-NEXT: %[[#rhs_adj:]] = extractvalue { i64, i64 } %[[#rhs]], 1
// LLVM-NEXT: %[[#adj_cmp:]] = icmp eq i64 %[[#lhs_adj]], %[[#rhs_adj]]
// LLVM-NEXT: %[[#tmp:]] = or i1 %[[#ptr_null]], %[[#adj_cmp]]
// LLVM-NEXT: %{{.+}} = and i1 %[[#tmp]], %[[#ptr_cmp]]

bool cmp_ne(void (Foo::*lhs)(int), void (Foo::*rhs)(int)) {
  return lhs != rhs;
}

// CHECK-LABEL: @_Z6cmp_neM3FooFviES1_
// CHECK: %{{.+}} = cir.cmp(ne, %{{.+}}, %{{.+}}) : !cir.method<!cir.func<(!s32i)> in !ty_Foo>, !cir.bool

// LLVM-LABEL: @_Z6cmp_neM3FooFviES1_
//      LLVM: %[[#lhs:]] = load { i64, i64 }, ptr %{{.+}}
// LLVM-NEXT: %[[#rhs:]] = load { i64, i64 }, ptr %{{.+}}
// LLVM-NEXT: %[[#lhs_ptr:]] = extractvalue { i64, i64 } %[[#lhs]], 0
// LLVM-NEXT: %[[#rhs_ptr:]] = extractvalue { i64, i64 } %[[#rhs]], 0
// LLVM-NEXT: %[[#ptr_cmp:]] = icmp ne i64 %[[#lhs_ptr]], %[[#rhs_ptr]]
// LLVM-NEXT: %[[#ptr_null:]] = icmp ne i64 %[[#lhs_ptr]], 0
// LLVM-NEXT: %[[#lhs_adj:]] = extractvalue { i64, i64 } %[[#lhs]], 1
// LLVM-NEXT: %[[#rhs_adj:]] = extractvalue { i64, i64 } %[[#rhs]], 1
// LLVM-NEXT: %[[#adj_cmp:]] = icmp ne i64 %[[#lhs_adj]], %[[#rhs_adj]]
// LLVM-NEXT: %[[#tmp:]] = and i1 %[[#ptr_null]], %[[#adj_cmp]]
// LLVM-NEXT: %{{.+}} = or i1 %[[#tmp]], %[[#ptr_cmp]]
