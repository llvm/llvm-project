// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t.before.log
// RUN: FileCheck %s --input-file=%t.before.log -check-prefix=CIR-BEFORE

struct Base {
  virtual ~Base();
};

struct Derived : Base {};

// CIR-BEFORE-DAG: !rec_Base = !cir.record
// CIR-BEFORE-DAG: !rec_Derived = !cir.record
// CIR-BEFORE-DAG: #dyn_cast_info__ZTI4Base__ZTI7Derived = #cir.dyn_cast_info<src_rtti = #cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>, dest_rtti = #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>, runtime_func = @__dynamic_cast, bad_cast_func = @__cxa_bad_cast, offset_hint = #cir.int<0> : !s64i>

Derived *ptr_cast(Base *b) {
  return dynamic_cast<Derived *>(b);
}

// CIR-BEFORE: cir.func dso_local @_Z8ptr_castP4Base
// CIR-BEFORE:   %{{.+}} = cir.dyn_cast ptr %{{.+}} : !cir.ptr<!rec_Base> -> !cir.ptr<!rec_Derived> #dyn_cast_info__ZTI4Base__ZTI7Derived
// CIR-BEFORE: }

Derived &ref_cast(Base &b) {
  return dynamic_cast<Derived &>(b);
}

// CIR-BEFORE: cir.func dso_local @_Z8ref_castR4Base
// CIR-BEFORE:   %{{.+}} = cir.dyn_cast ref %{{.+}} : !cir.ptr<!rec_Base> -> !cir.ptr<!rec_Derived> #dyn_cast_info__ZTI4Base__ZTI7Derived
// CIR-BEFORE: }
