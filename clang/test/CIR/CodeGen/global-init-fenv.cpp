// --- -ffp-exception-behavior=strict ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -ffp-exception-behavior=strict -emit-cir %s -o %t-strict.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-STRICT --input-file=%t-strict.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CONSTRAINED,LLVM-STRICT --input-file=%t-strict-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CONSTRAINED,LLVM-STRICT --input-file=%t-strict.ll %s

// --- -ffp-exception-behavior=maytrap ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -ffp-exception-behavior=maytrap -emit-cir %s -o %t-maytrap.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-MAYTRAP --input-file=%t-maytrap.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -ffp-exception-behavior=maytrap -emit-llvm %s -o %t-maytrap-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CONSTRAINED,LLVM-MAYTRAP --input-file=%t-maytrap-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti \
// RUN:   -ffp-exception-behavior=maytrap -emit-llvm %s -o %t-maytrap.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CONSTRAINED,LLVM-MAYTRAP --input-file=%t-maytrap.ll %s

// --- default FP environment (no constrained FP) ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -emit-cir %s -o %t-default.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-DEFAULT --input-file=%t-default.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -emit-llvm %s -o %t-default-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-DEFAULT --input-file=%t-default-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti \
// RUN:   -emit-llvm %s -o %t-default.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-DEFAULT --input-file=%t-default.ll %s

// A global initialized from two other globals declared in this module
// that are NOT constant is evaluated at runtime
float nc_x = 1.0f;
float nc_y = 10.0f;
float g_nonconst = nc_x / nc_y;

// CIR-LABEL:   cir.func {{.*}} @__cxx_global_var_init()
// CIR-STRICT-SAME:  attributes {strictfp}
// CIR-MAYTRAP-SAME: attributes {strictfp}
// CIR:         %[[G:.*]] = cir.get_global @g_nonconst : !cir.ptr<!cir.float>
// CIR:         %[[XP:.*]] = cir.get_global @nc_x : !cir.ptr<!cir.float>
// CIR:         %[[X:.*]] = cir.load {{.*}} %[[XP]]
// CIR:         %[[YP:.*]] = cir.get_global @nc_y : !cir.ptr<!cir.float>
// CIR:         %[[Y:.*]] = cir.load {{.*}} %[[YP]]
// CIR-STRICT:  %[[R:.*]] = cir.fdiv %[[X]], %[[Y]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: %[[R:.*]] = cir.fdiv %[[X]], %[[Y]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-DEFAULT: %[[R:.*]] = cir.fdiv %{{.*}}, %{{.*}} : !cir.float
// CIR-DEFAULT-NOT: fenv
// CIR:         cir.store {{.*}} %[[R]], %[[G]] : !cir.float, !cir.ptr<!cir.float>


// LLVM:         define {{.*}}void @__cxx_global_var_init()
// LLVM-CONSTRAINED-SAME: #[[ATTR:[0-9]+]]
// LLVM-STRICT:    call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP:   call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-DEFAULT:   fdiv float
// LLVM:           store float {{.*}}, ptr @g_nonconst


// A global initialized from two other globals declared in this module that are
// const (but NOT constexpr) is evaluated at runtime if we are in a constrained
// FP environment, otherwise the global is constant initialized at compile time.
const float c_x = 1.0f;
const float c_y = 10.0f;
float g_const = c_x / c_y;

// CIR-CONSTRAINED: cir.func {{.*}} @__cxx_global_var_init.1() attributes {strictfp}
// CIR-CONSTRAINED:   %[[G:.*]] = cir.get_global @g_const : !cir.ptr<!cir.float>
// CIR-CONSTRAINED:   %[[XP:.*]] = cir.get_global @_ZL3c_x : !cir.ptr<!cir.float>
// CIR-CONSTRAINED:   %[[X:.*]] = cir.load {{.*}} %[[XP]]
// CIR-CONSTRAINED:   %[[YP:.*]] = cir.get_global @_ZL3c_y : !cir.ptr<!cir.float>
// CIR-CONSTRAINED:   %[[Y:.*]] = cir.load {{.*}} %[[YP]]
// CIR-STRICT:        %[[R:.*]] = cir.fdiv %[[X]], %[[Y]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP:       %[[R:.*]] = cir.fdiv %[[X]], %[[Y]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-CONSTRAINED:   cir.store {{.*}} %[[R]], %[[G]] : !cir.float, !cir.ptr<!cir.float>

// LLVM-CONSTRAINED: define {{.*}}void @__cxx_global_var_init.1() #[[ATTR]]
// LLVM-STRICT:        call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP:       call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-CONSTRAINED:   store float {{.*}}, ptr @g_const

// Note, there are no DEFAULT checks here because there is no runtime evaluation.

// A global initialized from an expression using three other globals,
// two of which are constexpr and one which is not. Under a constrained FP
// environment none of the divisions are folded (folding would drop the
// rounding/exception behavior), so both divisions are evaluated at runtime.
constexpr float ce_x = 1.0f;
constexpr float ce_y = 10.0f;
float nc_z = 2.0f;
float g_mixed = ce_x / ce_y / nc_z;

// CIR-CONSTRAINED: cir.func {{.*}} @__cxx_global_var_init.2() attributes {strictfp}
// CIR-DEFAULT:     cir.func {{.*}} @__cxx_global_var_init.1()
// CIR:               %[[G:.*]] = cir.get_global @g_mixed : !cir.ptr<!cir.float>
// CIR:               %[[AP:.*]] = cir.get_global @_ZL4ce_x : !cir.ptr<!cir.float>
// CIR:               %[[A:.*]] = cir.load {{.*}} %[[AP]]
// CIR:               %[[BP:.*]] = cir.get_global @_ZL4ce_y : !cir.ptr<!cir.float>
// CIR:               %[[B:.*]] = cir.load {{.*}} %[[BP]]
// CIR-STRICT:        %[[R0:.*]] = cir.fdiv %[[A]], %[[B]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP:       %[[R0:.*]] = cir.fdiv %[[A]], %[[B]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-DEFAULT:       %[[R0:.*]] = cir.fdiv %[[A]], %[[B]] : !cir.float
// CIR-DEFAULT-NOT:     fenv
// CIR:               %[[CP:.*]] = cir.get_global @nc_z : !cir.ptr<!cir.float>
// CIR:               %[[C:.*]] = cir.load {{.*}} %[[CP]]
// CIR-STRICT:        %[[R1:.*]] = cir.fdiv %[[R0]], %[[C]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP:       %[[R1:.*]] = cir.fdiv %[[R0]], %[[C]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-DEFAULT:       %[[R1:.*]] = cir.fdiv %[[R0]], %[[C]] : !cir.float
// CIR-DEFAULT-NOT:     fenv
// CIR:               cir.store {{.*}} %[[R1]], %[[G]] : !cir.float, !cir.ptr<!cir.float>

// LLVM-CONSTRAINED: define {{.*}}void @__cxx_global_var_init.2() #[[ATTR]]
// LLVM-DEFAULT:     define {{.*}}void @__cxx_global_var_init.1()
// LLVM-STRICT:        call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT:        call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP:       call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-MAYTRAP:       call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-DEFAULT:       fdiv float
// LLVM:               store float {{.*}}, ptr @g_mixed


// LLVM-CONSTRAINED:         attributes #[[ATTR]] = {{.*}}strictfp

// LLVM-DEFAULT-NOT: constrained
// LLVM-DEFAULT-NOT: strictfp
