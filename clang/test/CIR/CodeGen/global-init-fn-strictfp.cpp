// --- -ffp-exception-behavior=strict ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -ffp-exception-behavior=strict -emit-cir %s -o %t-strict.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-EXCEPT --input-file=%t-strict.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-strict-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t-strict.ll %s

// --- -ffp-exception-behavior=maytrap ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -ffp-exception-behavior=maytrap -emit-cir %s -o %t-maytrap.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-EXCEPT --input-file=%t-maytrap.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -ffp-exception-behavior=maytrap -emit-llvm %s -o %t-maytrap-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-maytrap-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -ffp-exception-behavior=maytrap -emit-llvm %s -o %t-maytrap.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t-maytrap.ll %s

// --- -frounding-math ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -frounding-math -emit-cir %s -o %t-rounding.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-ROUNDING --input-file=%t-rounding.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -frounding-math -emit-llvm %s -o %t-rounding-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-rounding-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -frounding-math -emit-llvm %s -o %t-rounding.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t-rounding.ll %s

// --- default FP environment ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir %s -o %t-default.cir
// RUN: FileCheck --check-prefix=CIR-DEFAULT --input-file=%t-default.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-llvm %s -o %t-default-cir.ll
// RUN: FileCheck --check-prefix=LLVM-DEFAULT --input-file=%t-default-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o %t-default.ll
// RUN: FileCheck --check-prefix=LLVM-DEFAULT --input-file=%t-default.ll %s

// Classic codegen makes a function that has no FunctionDecl strictfp when the
// TU-wide default FP environment is constrained. LoweringPrepare creates such
// functions after CIRGen and reads the TU-wide default from
// #cir.lowering_options.

float ext();
struct S {
  S();
  ~S();
};

float g = ext() + 1.0f;
S arr[2];
thread_local float tl = ext() + 2.0f;
__attribute__((init_priority(200))) S prio;

void local() { static S sarr[2]; }

// CIR-EXCEPT: cir.lowering_options = #cir.lowering_options<{{.*}}no_trapping_math = false, ssp_buffer_size = 8>
// CIR-ROUNDING: cir.lowering_options = #cir.lowering_options<{{.*}}no_trapping_math = true, ssp_buffer_size = 8, rounding_math = true>

// CIR-DAG: cir.func internal private @__cxx_global_var_init() attributes {{{.*}}strictfp}
// CIR-DAG: cir.func internal private @__tls_init() attributes {{{.*}}strictfp}
// CIR-DAG: cir.func internal private @__cxx_global_array_dtor(%{{.*}}) attributes {{{.*}}strictfp}
// CIR-DAG: cir.func internal private @__cxx_global_array_dtor.1(%{{.*}}) attributes {{{.*}}strictfp}
// CIR-DAG: cir.func internal private @_GLOBAL__I_000200() attributes {{{.*}}strictfp}
// CIR-DAG: cir.func internal private @_GLOBAL__sub_I_{{.*}}() attributes {{{.*}}strictfp}

// CIR-DEFAULT: cir.lowering_options = #cir.lowering_options<{{.*}}no_trapping_math = true, ssp_buffer_size = 8>
// CIR-DEFAULT-NOT: attributes {{{.*}}strictfp

// The calls in these functions are strictfp too, as in classic codegen.
// ClangIR and classic codegen emit the functions in a different order, so the
// checks do not depend on it. Each line matches once in the module.

// LLVM-DAG: define internal void @__tls_init() #[[HELPER:[0-9]+]]
// LLVM-DAG:   call void @__cxx_global_var_init.2() #[[STRICT_CALL:[0-9]+]]
// LLVM-DAG: define internal void @__cxx_global_array_dtor(ptr noundef %{{.*}}) #[[HELPER]]
// LLVM-DAG: define internal void @_GLOBAL__I_000200() #[[HELPER]]
// LLVM-DAG:   call void @__cxx_global_var_init.3() #[[STRICT_CALL]]
// LLVM-DAG: define internal void @_GLOBAL__sub_I_{{.*}}() #[[HELPER]]
// LLVM-DAG:   call void @__cxx_global_var_init() #[[STRICT_CALL]]
// LLVM-DAG:   call void @__cxx_global_var_init.1() #[[STRICT_CALL]]

// The helper of the static local differs in the suffix that makes its name
// unique: LLVM counts renames across the module, LoweringPrepare per name.
// LLVMCIR-DAG: define internal void @__cxx_global_array_dtor.1(ptr noundef %{{.*}}) #[[HELPER]]
// OGCG-DAG: define internal void @__cxx_global_array_dtor.4(ptr noundef %{{.*}}) #[[HELPER]]

// LLVM-DAG: attributes #[[HELPER]] = { {{.*}}strictfp
// LLVM-DAG: attributes #[[STRICT_CALL]] = { strictfp }

// LLVM-DEFAULT: define internal void @_GLOBAL__sub_I_
// LLVM-DEFAULT-NOT: attributes #{{[0-9]+}} = {{.*}}strictfp
