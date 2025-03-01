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

struct Bar {
  void m4();
};

bool memfunc_to_bool(void (Foo::*func)(int)) {
  return func;
}

// CIR-LABEL: @_Z15memfunc_to_boolM3FooFviE
// CIR:   %{{.+}} = cir.cast(member_ptr_to_bool, %{{.+}} : !cir.method<!cir.func<(!s32i)> in !ty_Foo>), !cir.bool
// CIR: }

// LLVM-LABEL: @_Z15memfunc_to_boolM3FooFviE
//      LLVM:   %[[#memfunc:]] = load { i64, i64 }, ptr %{{.+}}
// LLVM-NEXT:   %[[#ptr:]] = extractvalue { i64, i64 } %[[#memfunc]], 0
// LLVM-NEXT:   %{{.+}} = icmp ne i64 %[[#ptr]], 0
//      LLVM: }

auto memfunc_reinterpret(void (Foo::*func)(int)) -> void (Bar::*)() {
  return reinterpret_cast<void (Bar::*)()>(func);
}

// CIR-LABEL: @_Z19memfunc_reinterpretM3FooFviE
// CIR:   %{{.+}} = cir.cast(bitcast, %{{.+}} : !cir.method<!cir.func<(!s32i)> in !ty_Foo>), !cir.method<!cir.func<()> in !ty_Bar>
// CIR: }

// LLVM-LABEL: @_Z19memfunc_reinterpretM3FooFviE
// LLVM-NEXT:   %[[#arg_slot:]] = alloca { i64, i64 }, i64 1
// LLVM-NEXT:   %[[#ret_slot:]] = alloca { i64, i64 }, i64 1
// LLVM-NEXT:   store { i64, i64 } %{{.+}}, ptr %[[#arg_slot]]
// LLVM-NEXT:   %[[#tmp:]] = load { i64, i64 }, ptr %[[#arg_slot]]
// LLVM-NEXT:   store { i64, i64 } %[[#tmp]], ptr %[[#ret_slot]]
// LLVM-NEXT:   %[[#ret:]] = load { i64, i64 }, ptr %[[#ret_slot]]
// LLVM-NEXT:   ret { i64, i64 } %[[#ret]]
// LLVM-NEXT: }

struct Base1 {
  int x;
  virtual void m1(int);
};

struct Base2 {
  int y;
  virtual void m2(int);
};

struct Derived : Base1, Base2 {
  virtual void m3(int);
};

using Base1MemFunc = void (Base1::*)(int);
using Base2MemFunc = void (Base2::*)(int);
using DerivedMemFunc = void (Derived::*)(int);

DerivedMemFunc base_to_derived_zero_offset(Base1MemFunc ptr) {
  return static_cast<DerivedMemFunc>(ptr);
}

// CIR-LABEL: @_Z27base_to_derived_zero_offsetM5Base1FviE
// CIR: %{{.+}} = cir.derived_method(%{{.+}} : !cir.method<!cir.func<(!s32i)> in !ty_Base1_>) [0] -> !cir.method<!cir.func<(!s32i)> in !ty_Derived>

// LLVM-LABEL: @_Z27base_to_derived_zero_offsetM5Base1FviE
// LLVM-NEXT:   %[[#arg_slot:]] = alloca { i64, i64 }, i64 1
// LLVM-NEXT:   %[[#ret_slot:]] = alloca { i64, i64 }, i64 1
// LLVM-NEXT:   store { i64, i64 } %{{.+}}, ptr %[[#arg_slot]]
// LLVM-NEXT:   %[[#tmp:]] = load { i64, i64 }, ptr %[[#arg_slot]]
// LLVM-NEXT:   store { i64, i64 } %[[#tmp]], ptr %[[#ret_slot]]
// LLVM-NEXT:   %[[#ret:]] = load { i64, i64 }, ptr %[[#ret_slot]]
// LLVM-NEXT:   ret { i64, i64 } %[[#ret]]
// LLVM-NEXT: }

DerivedMemFunc base_to_derived(Base2MemFunc ptr) {
  return static_cast<DerivedMemFunc>(ptr);
}

// CIR-LABEL: @_Z15base_to_derivedM5Base2FviE
// CIR: %{{.+}} = cir.derived_method(%{{.+}} : !cir.method<!cir.func<(!s32i)> in !ty_Base2_>) [16] -> !cir.method<!cir.func<(!s32i)> in !ty_Derived>

// LLVM-LABEL: @_Z15base_to_derivedM5Base2FviE
//      LLVM: %[[#arg:]] = load { i64, i64 }, ptr %{{.+}}
// LLVM-NEXT: %[[#adj:]] = extractvalue { i64, i64 } %[[#arg]], 1
// LLVM-NEXT: %[[#adj_adj:]] = add i64 %[[#adj]], 16
// LLVM-NEXT: %{{.+}} = insertvalue { i64, i64 } %[[#arg]], i64 %[[#adj_adj]], 1

Base1MemFunc derived_to_base_zero_offset(DerivedMemFunc ptr) {
  return static_cast<Base1MemFunc>(ptr);
}

// CIR-LABEL: @_Z27derived_to_base_zero_offsetM7DerivedFviE
// CIR: %{{.+}} = cir.base_method(%{{.+}} : !cir.method<!cir.func<(!s32i)> in !ty_Derived>) [0] -> !cir.method<!cir.func<(!s32i)> in !ty_Base1_>

// LLVM-LABEL: @_Z27derived_to_base_zero_offsetM7DerivedFviE
// LLVM-NEXT:   %[[#arg_slot:]] = alloca { i64, i64 }, i64 1
// LLVM-NEXT:   %[[#ret_slot:]] = alloca { i64, i64 }, i64 1
// LLVM-NEXT:   store { i64, i64 } %{{.+}}, ptr %[[#arg_slot]]
// LLVM-NEXT:   %[[#tmp:]] = load { i64, i64 }, ptr %[[#arg_slot]]
// LLVM-NEXT:   store { i64, i64 } %[[#tmp]], ptr %[[#ret_slot]]
// LLVM-NEXT:   %[[#ret:]] = load { i64, i64 }, ptr %[[#ret_slot]]
// LLVM-NEXT:   ret { i64, i64 } %[[#ret]]
// LLVM-NEXT: }

Base2MemFunc derived_to_base(DerivedMemFunc ptr) {
  return static_cast<Base2MemFunc>(ptr);
}

// CIR-LABEL: @_Z15derived_to_baseM7DerivedFviE
// CIR: %{{.+}} = cir.base_method(%{{.+}} : !cir.method<!cir.func<(!s32i)> in !ty_Derived>) [16] -> !cir.method<!cir.func<(!s32i)> in !ty_Base2_>

// LLVM-LABEL: @_Z15derived_to_baseM7DerivedFviE
//      LLVM: %[[#arg:]] = load { i64, i64 }, ptr %{{.+}}
// LLVM-NEXT: %[[#adj:]] = extractvalue { i64, i64 } %[[#arg]], 1
// LLVM-NEXT: %[[#adj_adj:]] = sub i64 %[[#adj]], 16
// LLVM-NEXT: %{{.+}} = insertvalue { i64, i64 } %[[#arg]], i64 %[[#adj_adj]], 1
