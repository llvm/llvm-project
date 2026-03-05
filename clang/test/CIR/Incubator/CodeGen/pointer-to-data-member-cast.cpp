// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

struct Base1 {
  int base1_data;
};

struct Base2 {
  int base2_data;
};

struct Derived : Base1, Base2 {
  int derived_data;
};

// CIR-LABEL:  @_Z15base_to_derivedM5Base2i
// LLVM-LABEL: @_Z15base_to_derivedM5Base2i
auto base_to_derived(int Base2::*ptr) -> int Derived::* {
  return ptr;
  // CIR: %{{.+}} = cir.derived_data_member %{{.+}} : !cir.data_member<!s32i in !rec_Base2> [4] -> !cir.data_member<!s32i in !rec_Derived>

  //      LLVM: %[[#src:]] = load i64, ptr %{{.+}}
  // LLVM-NEXT: %[[#is_null:]] = icmp eq i64 %[[#src]], -1
  // LLVM-NEXT: %[[#adjusted:]] = add i64 %[[#src]], 4
  // LLVM-NEXT: %{{.+}} = select i1 %[[#is_null]], i64 -1, i64 %[[#adjusted]]
}

// CIR-LABEL:  @_Z15derived_to_baseM7Derivedi
// LLVM-LABEL: @_Z15derived_to_baseM7Derivedi
auto derived_to_base(int Derived::*ptr) -> int Base2::* {
  return static_cast<int Base2::*>(ptr);
  // CIR: %{{.+}} = cir.base_data_member %{{.+}} : !cir.data_member<!s32i in !rec_Derived> [4] -> !cir.data_member<!s32i in !rec_Base2>

  //      LLVM: %[[#src:]] = load i64, ptr %{{.+}}
  // LLVM-NEXT: %[[#is_null:]] = icmp eq i64 %[[#src]], -1
  // LLVM-NEXT: %[[#adjusted:]] = sub i64 %[[#src]], 4
  // LLVM-NEXT: %{{.+}} = select i1 %[[#is_null]], i64 -1, i64 %[[#adjusted]]
}

// CIR-LABEL:  @_Z27base_to_derived_zero_offsetM5Base1i
// LLVM-LABEL: @_Z27base_to_derived_zero_offsetM5Base1i
auto base_to_derived_zero_offset(int Base1::*ptr) -> int Derived::* {
  return ptr;
  // CIR: %{{.+}} = cir.derived_data_member %{{.+}} : !cir.data_member<!s32i in !rec_Base1> [0] -> !cir.data_member<!s32i in !rec_Derived>

  // No LLVM instructions emitted for performing a zero-offset cast.
  // LLVM-NEXT: %[[#src_slot:]] = alloca i64, i64 1
  // LLVM-NEXT: %[[#ret_slot:]] = alloca i64, i64 1
  // LLVM-NEXT: store i64 %{{.+}}, ptr %[[#src_slot]]
  // LLVM-NEXT: %[[#temp:]] = load i64, ptr %[[#src_slot]]
  // LLVM-NEXT: store i64 %[[#temp]], ptr %[[#ret_slot]]
  // LLVM-NEXT: %[[#ret:]] = load i64, ptr %[[#ret_slot]]
  // LLVM-NEXT: ret i64 %[[#ret]]
}

// CIR-LABEL:  @_Z27derived_to_base_zero_offsetM7Derivedi
// LLVM-LABEL: @_Z27derived_to_base_zero_offsetM7Derivedi
auto derived_to_base_zero_offset(int Derived::*ptr) -> int Base1::* {
  return static_cast<int Base1::*>(ptr);
  // CIR: %{{.+}} = cir.base_data_member %{{.+}} : !cir.data_member<!s32i in !rec_Derived> [0] -> !cir.data_member<!s32i in !rec_Base1>

  // No LLVM instructions emitted for performing a zero-offset cast.
  // LLVM-NEXT: %[[#src_slot:]] = alloca i64, i64 1
  // LLVM-NEXT: %[[#ret_slot:]] = alloca i64, i64 1
  // LLVM-NEXT: store i64 %{{.+}}, ptr %[[#src_slot]]
  // LLVM-NEXT: %[[#temp:]] = load i64, ptr %[[#src_slot]]
  // LLVM-NEXT: store i64 %[[#temp]], ptr %[[#ret_slot]]
  // LLVM-NEXT: %[[#ret:]] = load i64, ptr %[[#ret_slot]]
  // LLVM-NEXT: ret i64 %[[#ret]]
}

struct Foo {
  int a;
};

struct Bar {
  int a;
};

bool to_bool(int Foo::*x) {
  return x;
}

// CIR-LABEL: @_Z7to_boolM3Fooi
//      CIR:   %[[#x:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-NEXT:   %{{.+}} = cir.cast member_ptr_to_bool %[[#x]] : !cir.data_member<!s32i in !rec_Foo> -> !cir.bool
//      CIR: }

auto bitcast(int Foo::*x) {
  return reinterpret_cast<int Bar::*>(x);
}

// CIR-LABEL: @_Z7bitcastM3Fooi
//      CIR:   %[[#x:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-NEXT:   %{{.+}} = cir.cast bitcast %[[#x]] : !cir.data_member<!s32i in !rec_Foo> -> !cir.data_member<!s32i in !rec_Bar>
//      CIR: }
