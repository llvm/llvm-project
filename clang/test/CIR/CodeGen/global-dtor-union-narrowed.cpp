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

// With the layout-driven const_record path, @g keeps its logical !rec_S
// type, so no pointer narrowing/bitcast is needed.

// CIR: !rec_S = !cir.struct<"S"
// CIR: cir.global external @g = ctor : !rec_S {
// CIR:   %[[GET_GLOB:.+]] = cir.get_global @g : !cir.ptr<!rec_S>
// CIR:   cir.call @_ZN1SC2Ev(%[[GET_GLOB]])
// CIR: }

// LLVM: @g = global %struct.S zeroinitializer
// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   call void @_ZN1SC2Ev(ptr {{.*}}@g)
// LLVM:   call void @__cxa_atexit(ptr @_ZN1SD2Ev, ptr @g, ptr @__dso_handle)

// OGCG: @g = global { { { [16 x i8] } } } zeroinitializer
// OGCG: define internal void @__cxx_global_var_init()
// OGCG:   call i32 @__cxa_atexit(ptr @_ZN1SD2Ev, ptr @g, ptr @__dso_handle)
