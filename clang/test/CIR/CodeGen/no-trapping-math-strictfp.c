// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// A pragma can make a function observe FP exceptions even though the TU-wide
// default ignores them. Such a function is strictfp and does not get
// "no-trapping-math". Classic codegen still adds "no-trapping-math"="true"
// from the TU-wide default, which contradicts the strict constrained
// intrinsics in the function body.

float plain(float a, float b) { return a + b; }

float except_on(float a, float b) {
#pragma float_control(except, on)
  return a + b;
}

#pragma STDC FENV_ACCESS ON
float fenv_access(float a, float b) { return a + b; }

// CIR: cir.lowering_options = #cir.lowering_options<{{.*}}no_trapping_math = true
// CIR: cir.func {{.*}}@except_on({{.*}}) -> {{.*}} attributes {{{.*}}strictfp}
// CIR: cir.func {{.*}}@fenv_access({{.*}}) -> {{.*}} attributes {{{.*}}strictfp}

// LLVM: define {{.*}}float @plain({{.*}}) #[[PLAIN:[0-9]+]]
// LLVM: define {{.*}}float @except_on({{.*}}) #[[STRICT:[0-9]+]]
// LLVM:   call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM: define {{.*}}float @fenv_access({{.*}}) #[[STRICT]]
// LLVM:   call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM: attributes #[[PLAIN]] = { {{.*}}"no-trapping-math"="true"
// String attributes are sorted, so "no-trapping-math" would precede
// "stack-protector-buffer-size".
// LLVM: attributes #[[STRICT]] = { {{[^"]*}}strictfp "stack-protector-buffer-size"="8"

// OGCG: define {{.*}}float @plain({{.*}}) #[[PLAIN:[0-9]+]]
// OGCG: define {{.*}}float @except_on({{.*}}) #[[STRICT:[0-9]+]]
// OGCG: define {{.*}}float @fenv_access({{.*}}) #[[STRICT]]
// OGCG: attributes #[[PLAIN]] = { {{.*}}"no-trapping-math"="true"
// OGCG: attributes #[[STRICT]] = { {{.*}}strictfp {{.*}}"no-trapping-math"="true"
