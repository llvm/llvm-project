// Verify that C++ dynamic global initializers run under a constrained
// floating-point environment when the default rounding/exception mode is
// non-standard. Each dynamic initializer is emitted into the corresponding
// global's ctor region with #cir.fenv attached to its floating-point
// operations, and the generated __cxx_global_var_init function is marked
// strictfp. Lowering to LLVM IR should match classic codegen.
//
// The floating-point behavior of an initializer depends on the storage
// duration of the object being initialized (C99 F.7.4 / [expr.const]). A
// constant-expression initializer of an object with static storage duration is
// evaluated at translation time with the default rounding mode and raises no
// exceptions, so it is folded. A non-constant (dynamic) initializer is
// evaluated during execution and is therefore subject to the operative
// floating-point environment.

// --- -ffp-exception-behavior=strict ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -ffp-exception-behavior=strict -emit-cir %s -o %t-strict.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-STRICT --input-file=%t-strict.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT --input-file=%t-strict-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT --input-file=%t-strict.ll %s

// --- -ffp-exception-behavior=maytrap ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -ffp-exception-behavior=maytrap -emit-cir %s -o %t-maytrap.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-MAYTRAP --input-file=%t-maytrap.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -ffp-exception-behavior=maytrap -emit-llvm %s -o %t-maytrap-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-MAYTRAP --input-file=%t-maytrap-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti \
// RUN:   -ffp-exception-behavior=maytrap -emit-llvm %s -o %t-maytrap.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-MAYTRAP --input-file=%t-maytrap.ll %s

// --- default FP environment (no constrained FP) ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -emit-cir %s -o %t-default.cir
// RUN: FileCheck --check-prefixes=CIR-DEFAULT --input-file=%t-default.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-rtti -fclangir \
// RUN:   -emit-llvm %s -o %t-default-cir.ll
// RUN: FileCheck --check-prefixes=LLVM-DEFAULT --input-file=%t-default-cir.ll %s

//===----------------------------------------------------------------------===//
// Case 1: a global initialized from two other globals declared in this module
// that are NOT constant. Their values are unknown at translation time, so the
// initializer is always dynamic.
//===----------------------------------------------------------------------===//
float nc_x = 1.0f;
float nc_y = 10.0f;
float g_nonconst = nc_x / nc_y;

//===----------------------------------------------------------------------===//
// Case 2: a global initialized from two other globals declared in this module
// that are const (but NOT constexpr). A const float is not usable in a constant
// expression, so under a constrained FP environment the division is evaluated
// at runtime; in the default environment it is folded to a static constant.
//===----------------------------------------------------------------------===//
const float c_x = 1.0f;
const float c_y = 10.0f;
float g_const = c_x / c_y;

//===----------------------------------------------------------------------===//
// Case 3: a global initialized from an expression using three other globals,
// two of which are constexpr and one which is not. Under a constrained FP
// environment none of the divisions are folded (folding would drop the
// rounding/exception behavior), so both divisions are evaluated at runtime.
//===----------------------------------------------------------------------===//
constexpr float ce_x = 1.0f;
constexpr float ce_y = 10.0f;
float nc_z = 2.0f;
float g_mixed = ce_x / ce_y / nc_z;

//===----------------------------------------------------------------------===//
// Constrained FP (strict / maytrap): every global is initialized dynamically.
// Each __cxx_global_var_init function is marked strictfp, and each floating
// point division carries a #cir.fenv attribute (strict_except differs between
// the strict and maytrap exception modes). 1.0f / 10.0f is inexact, so the
// division requires rounding and raises the inexact exception.
//===----------------------------------------------------------------------===//

// Case 1: g_nonconst = nc_x / nc_y
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
// CIR:         cir.store {{.*}} %[[R]], %[[G]] : !cir.float, !cir.ptr<!cir.float>

// Case 2: g_const = c_x / c_y  (const operands loaded, not folded)
// CIR-LABEL:   cir.func {{.*}} @__cxx_global_var_init.1()
// CIR-STRICT-SAME:  attributes {strictfp}
// CIR-MAYTRAP-SAME: attributes {strictfp}
// CIR:         %[[G:.*]] = cir.get_global @g_const : !cir.ptr<!cir.float>
// CIR:         %[[XP:.*]] = cir.get_global @_ZL3c_x : !cir.ptr<!cir.float>
// CIR:         %[[X:.*]] = cir.load {{.*}} %[[XP]]
// CIR:         %[[YP:.*]] = cir.get_global @_ZL3c_y : !cir.ptr<!cir.float>
// CIR:         %[[Y:.*]] = cir.load {{.*}} %[[YP]]
// CIR-STRICT:  %[[R:.*]] = cir.fdiv %[[X]], %[[Y]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: %[[R:.*]] = cir.fdiv %[[X]], %[[Y]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR:         cir.store {{.*}} %[[R]], %[[G]] : !cir.float, !cir.ptr<!cir.float>

// Case 3: g_mixed = ce_x / ce_y / nc_z  (all operands loaded, two divisions)
// CIR-LABEL:   cir.func {{.*}} @__cxx_global_var_init.2()
// CIR-STRICT-SAME:  attributes {strictfp}
// CIR-MAYTRAP-SAME: attributes {strictfp}
// CIR:         %[[G:.*]] = cir.get_global @g_mixed : !cir.ptr<!cir.float>
// CIR:         %[[AP:.*]] = cir.get_global @_ZL4ce_x : !cir.ptr<!cir.float>
// CIR:         %[[A:.*]] = cir.load {{.*}} %[[AP]]
// CIR:         %[[BP:.*]] = cir.get_global @_ZL4ce_y : !cir.ptr<!cir.float>
// CIR:         %[[B:.*]] = cir.load {{.*}} %[[BP]]
// CIR-STRICT:  %[[R0:.*]] = cir.fdiv %[[A]], %[[B]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: %[[R0:.*]] = cir.fdiv %[[A]], %[[B]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR:         %[[CP:.*]] = cir.get_global @nc_z : !cir.ptr<!cir.float>
// CIR:         %[[C:.*]] = cir.load {{.*}} %[[CP]]
// CIR-STRICT:  %[[R1:.*]] = cir.fdiv %[[R0]], %[[C]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: %[[R1:.*]] = cir.fdiv %[[R0]], %[[C]] : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR:         cir.store {{.*}} %[[R1]], %[[G]] : !cir.float, !cir.ptr<!cir.float>

//===----------------------------------------------------------------------===//
// LLVM lowering (strict / maytrap): the same via -fclangir and classic codegen.
// Each initializer function is strictfp and each division becomes a constrained
// fdiv intrinsic. (The operand values differ between pipelines because CIR
// loads const/constexpr operands from memory while classic codegen substitutes
// their constant values, so the operands are wildcarded here.)
//===----------------------------------------------------------------------===//

// Case 1: g_nonconst
// LLVM:         define {{.*}}void @__cxx_global_var_init() #[[ATTR:[0-9]+]]
// LLVM-STRICT:  call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM:         store float {{.*}}, ptr @g_nonconst

// Case 2: g_const
// LLVM:         define {{.*}}void @__cxx_global_var_init.1() #[[ATTR]]
// LLVM-STRICT:  call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM:         store float {{.*}}, ptr @g_const

// Case 3: g_mixed (two constrained divisions)
// LLVM:         define {{.*}}void @__cxx_global_var_init.2() #[[ATTR]]
// LLVM-STRICT:  call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT:  call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.fdiv.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM:         store float {{.*}}, ptr @g_mixed

// LLVM:         attributes #[[ATTR]] = {{.*}}strictfp

//===----------------------------------------------------------------------===//
// Default FP environment: g_nonconst and g_mixed are still dynamic (they use
// non-constant operands), but neither init function is strictfp and no fenv
// attribute is attached. g_const IS a constant expression here, so it is folded
// to a static constant and needs no initializer function -- which is why
// g_mixed's initializer is numbered .1 (not .2): g_const never gets one.
//===----------------------------------------------------------------------===//

// Case 1 (default): g_nonconst, plain fdiv, no strictfp / no fenv.
// CIR-DEFAULT-LABEL: cir.func internal private @__cxx_global_var_init() {
// CIR-DEFAULT:   %[[G:.*]] = cir.get_global @g_nonconst : !cir.ptr<!cir.float>
// CIR-DEFAULT:   %[[R:.*]] = cir.fdiv %{{.*}}, %{{.*}} : !cir.float loc
// CIR-DEFAULT:   cir.store {{.*}} %[[R]], %[[G]] : !cir.float, !cir.ptr<!cir.float>

// Case 2 (default): g_const folded to a static constant (1.0f / 10.0f = 0.1f).
// CIR-DEFAULT: cir.global external @g_const = #cir.fp<1.000000e-01> : !cir.float

// Case 3 (default): g_mixed, ce_x/ce_y still loaded and divided at runtime,
// then divided by nc_z. Two plain fdivs, no strictfp / no fenv.
// CIR-DEFAULT-LABEL: cir.func internal private @__cxx_global_var_init.1() {
// CIR-DEFAULT:   %[[G2:.*]] = cir.get_global @g_mixed : !cir.ptr<!cir.float>
// CIR-DEFAULT:   %[[R0:.*]] = cir.fdiv %{{.*}}, %{{.*}} : !cir.float loc
// CIR-DEFAULT:   %[[R1:.*]] = cir.fdiv %[[R0]], %{{.*}} : !cir.float loc
// CIR-DEFAULT:   cir.store {{.*}} %[[R1]], %[[G2]] : !cir.float, !cir.ptr<!cir.float>

// LLVM (default): folded constant, plain fdivs, no constrained intrinsics.
// LLVM-DEFAULT: @g_const = {{.*}}global float 1.000000e-01
// LLVM-DEFAULT: define {{.*}}void @__cxx_global_var_init()
// LLVM-DEFAULT: fdiv float
// LLVM-DEFAULT: store float {{.*}}, ptr @g_nonconst
// LLVM-DEFAULT: define {{.*}}void @__cxx_global_var_init.1()
// LLVM-DEFAULT: store float {{.*}}, ptr @g_mixed
// LLVM-DEFAULT-NOT: constrained
// LLVM-DEFAULT-NOT: strictfp
