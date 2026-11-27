// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions \
// RUN:   -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

// With a strict TU-wide default the global initializer is strictfp too.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions \
// RUN:   -ffp-exception-behavior=strict -fclangir -emit-llvm %s -o %t-strict-cir.ll
// RUN: FileCheck --input-file=%t-strict-cir.ll %s \
// RUN:   -check-prefixes=LLVM-STRICT,LLVMCIR-STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --input-file=%t-strict.ll %s \
// RUN:   -check-prefixes=LLVM-STRICT,OGCG-STRICT

// Calls in a strictfp function are strictfp in LLVM IR, including invokes,
// intrinsic calls and the calls that EH lowering creates.

float ext();
struct C { C(); };
struct D { ~D(); };

float g = ext() + 1.0f;

// LLVM-LABEL: define internal void @__cxx_global_var_init()
// LLVM:         call noundef float @_Z3extv(){{$}}

// Classic codegen marks a call it emits for a call expression only when the
// enclosing declaration is a strictfp function, so it leaves this one unmarked
// beside a constrained intrinsic. ClangIR marks it, as LangRef asks.
// LLVM-STRICT-LABEL: define internal void @__cxx_global_var_init()
// LLVM-STRICT-SAME:    #[[INIT_FN:[0-9]+]]
// LLVMCIR-STRICT:      call noundef float @_Z3extv() #[[STRICT_CALL:[0-9]+]]
// LLVMCIR-STRICT:      call float @llvm.experimental.constrained.fadd.f32({{.*}}) #[[STRICT_CALL]]
// OGCG-STRICT:         call noundef float @_Z3extv(){{$}}
// OGCG-STRICT:         call float @llvm.experimental.constrained.fadd.f32({{.*}}) #[[STRICT_CALL:[0-9]+]]

C c;

// LLVM-LABEL: define internal void @__cxx_global_var_init.1()
// LLVM:         call void @_ZN1CC1Ev(ptr {{.*}}@c){{$}}

// The same holds for a construct expression.
// LLVM-STRICT-LABEL: define internal void @__cxx_global_var_init.1()
// LLVM-STRICT-SAME:    #[[INIT_FN]]
// LLVMCIR-STRICT:      call void @_ZN1CC1Ev(ptr {{.*}}@c) #[[STRICT_CALL]]
// OGCG-STRICT:         call void @_ZN1CC1Ev(ptr {{.*}}@c){{$}}

float plain() {
  D d;
  return ext();
}

// LLVM-LABEL: define {{.*}}float @_Z5plainv()
// LLVM:         invoke noundef float @_Z3extv(){{$}}
// LLVM:         call void @_ZN1DD1Ev(ptr {{.*}}) #[[NOUNWIND_CALL:[0-9]+]]

float strict() {
#pragma STDC FENV_ACCESS ON
  D d;
  try {
    return ext();
  } catch (...) {
    return 0;
  }
}

// Classic codegen also marks __cxa_begin_catch nounwind and invokes
// __cxa_end_catch, which is unrelated to strictfp.
// LLVM-LABEL: define {{.*}}float @_Z6strictv()
// LLVM-SAME:    #[[STRICT_FN:[0-9]+]]
// LLVM:         invoke noundef float @_Z3extv() #[[STRICT_CALL:[0-9]+]]
// LLVMCIR:      call ptr @__cxa_begin_catch(ptr %{{.*}}) #[[STRICT_CALL]]
// OGCG:         call ptr @__cxa_begin_catch(ptr %{{.*}}) #[[STRICT_NOUNWIND_CALL:[0-9]+]]
// LLVM:         call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}) #[[STRICT_CALL]]
// LLVM:         {{call|invoke}} void @__cxa_end_catch() #[[STRICT_CALL]]
// LLVMCIR:      call void @_ZN1DD1Ev(ptr {{.*}}) #[[STRICT_NOUNWIND_CALL:[0-9]+]]
// OGCG:         call void @_ZN1DD1Ev(ptr {{.*}}) #[[STRICT_NOUNWIND_CALL]]
// LLVM:         call void @llvm.trap() #[[STRICT_CALL]]

// LLVM-DAG: attributes #[[STRICT_FN]] = { {{.*}}strictfp
// LLVM-DAG: attributes #[[NOUNWIND_CALL]] = { nounwind }
// LLVM-DAG: attributes #[[STRICT_CALL]] = { strictfp }
// LLVM-DAG: attributes #[[STRICT_NOUNWIND_CALL]] = { nounwind strictfp }

// LLVM-STRICT-DAG: attributes #[[INIT_FN]] = { {{.*}}strictfp
// LLVM-STRICT-DAG: attributes #[[STRICT_CALL]] = { strictfp }
