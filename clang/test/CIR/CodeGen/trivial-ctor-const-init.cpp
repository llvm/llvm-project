// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct StructWithDefaultCtor {
  int n;
};

StructWithDefaultCtor defCtor = StructWithDefaultCtor();

// CIR: cir.global {{.*}} @defCtor = #cir.zero : !rec_StructWithDefaultCtor
// LLVM: @defCtor = global %struct.StructWithDefaultCtor zeroinitializer
// OGCG: @defCtor = global %struct.StructWithDefaultCtor zeroinitializer

struct StructWithCtorArg {
  double value;
  StructWithCtorArg(const double& x) : value(x) {}
};

StructWithCtorArg withArg = 0.0;

// CIR: cir.global {{.*}} @withArg = #cir.zero : !rec_StructWithCtorArg
// LLVM: @withArg = global %struct.StructWithCtorArg zeroinitializer
// OGCG: @withArg = global %struct.StructWithCtorArg zeroinitializer

// CIR: cir.func {{.*}} @__cxx_global_var_init()
// CIR:   %[[WITH_ARG:.*]] = cir.get_global @withArg : !cir.ptr<!rec_StructWithCtorArg>
// CIR:   cir.scope {
// CIR:     %[[TMP0:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["ref.tmp0"]
// CIR:     %[[ZERO:.*]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CIR:     cir.store{{.*}} %[[ZERO]], %[[TMP0]] : !cir.double, !cir.ptr<!cir.double>
// CIR:     cir.call @_ZN17StructWithCtorArgC1ERKd(%[[WITH_ARG]], %[[TMP0]]) : (!cir.ptr<!rec_StructWithCtorArg>, !cir.ptr<!cir.double>) -> ()
// CIR:   }

// LLVM: define {{.*}} void @__cxx_global_var_init()
// LLVM:   %[[TMP0:.*]] = alloca double
// LLVM:   store double 0.000000e+00, ptr %[[TMP0]]
// LLVM:   call void @_ZN17StructWithCtorArgC1ERKd(ptr @withArg, ptr %[[TMP0]])

// OGCG: define {{.*}} void @__cxx_global_var_init()
// OGCG:   %[[TMP0:.*]] = alloca double
// OGCG:   store double 0.000000e+00, ptr %[[TMP0]]
// OGCG:   call void @_ZN17StructWithCtorArgC1ERKd(ptr {{.*}} @withArg, ptr {{.*}} %[[TMP0]])
