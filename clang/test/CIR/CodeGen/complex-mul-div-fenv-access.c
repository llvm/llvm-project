// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=FULL \
// RUN:   --implicit-check-not="{{= f(add|sub|mul|div|cmp|pext|ptrunc) }}"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=FULL

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=improved -fclangir -emit-llvm %s -o %t-improved-cir.ll
// RUN: FileCheck --input-file=%t-improved-cir.ll %s --check-prefix=IMPROVED \
// RUN:   --implicit-check-not="{{= f(add|sub|mul|div|cmp|pext|ptrunc) }}"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=improved -emit-llvm %s -o %t-improved.ll
// RUN: FileCheck --input-file=%t-improved.ll %s --check-prefix=IMPROVED

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=promoted -fclangir -emit-llvm %s -o %t-promoted-cir.ll
// RUN: FileCheck --input-file=%t-promoted-cir.ll %s --check-prefix=PROMOTED \
// RUN:   --implicit-check-not="{{= f(add|sub|mul|div|cmp|pext|ptrunc) }}"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -complex-range=promoted -emit-llvm %s -o %t-promoted.ll
// RUN: FileCheck --input-file=%t-promoted.ll %s --check-prefix=PROMOTED

// Complex multiplication and division under FENV_ACCESS. The arithmetic,
// comparisons and conversions of the expansion are constrained intrinsics.
// ClangIR and classic codegen share the check lines, so both emit the same
// intrinsics with the same metadata. The implicit check rejects an
// unconstrained instruction in the ClangIR output.
//
// The check lines end before the call-site attributes.
// FIXME: ClangIR does not mark the calls in a strictfp function `strictfp`.
// FIXME: ClangIR omits `nounwind` on the library calls and `noundef` on their
// arguments.

#pragma STDC FENV_ACCESS ON

_Complex float a, b, c;

void mul(void) { c = a * b; }

// With the full range, a NaN check of the product guards a call to the runtime
// library.

// FULL-LABEL: define{{.*}} void @mul()
// FULL-COUNT-6: call float @llvm.experimental.constrained.f{{mul|sub|add}}.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// FULL-COUNT-2: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
// FULL: call <2 x float> @__mulsc3(

void div(void) { c = a / b; }

// With the full range, division is a call to the runtime library.

// FULL-LABEL: define{{.*}} void @div()
// FULL: call <2 x float> @__divsc3(

// With the improved range, division uses Smith's algorithm. The range check
// is a quiet comparison.

// IMPROVED-LABEL: define{{.*}} void @div()
// IMPROVED-COUNT-2: call float @llvm.fabs.f32(float %{{.*}})
// IMPROVED: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
// IMPROVED-COUNT-18: call float @llvm.experimental.constrained.f{{mul|add|sub|div}}.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

// With the promoted range, division is done in a wider type. The conversions
// to and from that type are part of the division and take its environment.

// PROMOTED-LABEL: define{{.*}} void @div()
// PROMOTED-COUNT-4: call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
// PROMOTED-COUNT-11: call double @llvm.experimental.constrained.f{{mul|add|sub|div}}.f64(double %{{.*}}, double %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// PROMOTED-COUNT-2: call float @llvm.experimental.constrained.fptrunc.f32.f64(double %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
