// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t-before.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  struct Rep {
    union {
      char shortbuf[16];
      struct { long sz; char *p; } longrep;
    };
    constexpr Rep() : shortbuf{} {}
  } r;

  constexpr S() : r() {}
  ~S() {}
};

S g;

// The dtor region must bitcast `@g`'s narrowed pointer back to `!rec_S`
// before the dtor call.

// CIR: !rec_S = !cir.record<struct "S"
// CIR: cir.global external @g = #cir.zero : ![[NARROW_TY:rec_anon_struct[0-9]*]] dtor {
// CIR:   %[[ADDR:.+]] = cir.get_global @g : !cir.ptr<![[NARROW_TY]]>
// CIR:   %[[CAST:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<![[NARROW_TY]]> -> !cir.ptr<!rec_S>
// CIR:   cir.call @_ZN1SD2Ev(%[[CAST]]) : (!cir.ptr<!rec_S>) -> ()
// CIR: }

// LLVM: @g = global { { { [16 x i8] } } } zeroinitializer
// LLVM: define internal void @__cxx_global_array_dtor(ptr noundef %[[A0:.*]])
// LLVM:   call void @_ZN1SD2Ev(ptr %[[A0]])
// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   call void @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr @g, ptr @__dso_handle)

// OGCG: @g = global { { { [16 x i8] } } } zeroinitializer
// OGCG: define internal void @__cxx_global_var_init()
// OGCG:   call i32 @__cxa_atexit(ptr @_ZN1SD2Ev, ptr @g, ptr @__dso_handle)
