// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-host \
// RUN:   -Wno-unevaluated-expression -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR-HOST --input-file=%t.cir
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-host \
// RUN:   -Wno-unevaluated-expression -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefixes=LLVM-HOST,CIR-LLVM-HOST --input-file=%t-cir.ll
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-host \
// RUN:   -Wno-unevaluated-expression -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefixes=LLVM-HOST,OGCG-HOST --input-file=%t.ll
// RUN: %clang_cc1 -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device \
// RUN:   -Wno-unevaluated-expression -fclangir -emit-cir %s -o %t-dev.cir
// RUN: FileCheck %s --check-prefix=CIR-DEV --input-file=%t-dev.cir
// RUN: %clang_cc1 -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device \
// RUN:   -Wno-unevaluated-expression -fclangir -emit-llvm %s -o %t-dev-cir.ll
// RUN: FileCheck %s --check-prefixes=LLVM-DEV,CIR-LLVM-DEV --input-file=%t-dev-cir.ll
// RUN: %clang_cc1 -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device \
// RUN:   -Wno-unevaluated-expression -emit-llvm %s -o %t-dev.ll
// RUN: FileCheck %s --check-prefixes=LLVM-DEV,OGCG-DEV --input-file=%t-dev.ll

// __builtin_sycl_unique_stable_name evaluated at run time emits a private
// "__usn_str" global in the global constant address space, cast to the type
// of the expression. Constant-evaluated uses are folded to a string literal.

// Required by sycl_kernel_entry_point semantics.
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

void use(const char *);

// A global initializer is constant-folded to a string literal.
const char *global_name = __builtin_sycl_unique_stable_name(int);

// CIR-HOST: cir.global "private" constant cir_private dso_local @[[GLOBAL_STR:[^ ]+]] = #cir.const_array<"_ZTSi" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> align(1)
// CIR-HOST: cir.global external @global_name = #cir.global_view<@[[GLOBAL_STR]]> : !cir.ptr<!s8i> align(8)
// CIR-HOST: cir.global "private" constant cir_private dso_local @[[LAMBDA:__usn_str]] = #cir.const_array<"_ZTSZ9host_onlyvEUlvE_" : !cir.array<!s8i x 22>, trailing_zeros> : !cir.array<!s8i x 23> align(1)
// CIR-HOST: cir.global "private" constant cir_private dso_local @[[INT:__usn_str\.[0-9]+]] = #cir.const_array<"_ZTSi" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> align(1)
// CIR-HOST: cir.global "private" constant cir_private dso_local @[[CONSTEXPR_STR:[^ ]+]] = #cir.const_array<"_ZTSl" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> align(1)
// CIR-HOST: cir.global "private" constant cir_private dso_local @[[TPL_INT:__usn_str\.[0-9]+]] = #cir.const_array<"_ZTSi" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> align(1)
// CIR-HOST: cir.global "private" constant cir_private dso_local @[[TPL_LAMBDA:__usn_str\.[0-9]+]] = #cir.const_array<"_ZTSZ9host_onlyvEUlvE_" : !cir.array<!s8i x 22>, trailing_zeros> : !cir.array<!s8i x 23> align(1)

// CIR-LLVM-HOST: @[[GLOBAL_STR:[^ ]+]] = private constant [6 x i8] c"_ZTSi\00", align 1
// CIR-LLVM-HOST: @global_name = global ptr @[[GLOBAL_STR]], align 8
// CIR-LLVM-HOST: @[[LAMBDA:__usn_str]] = private constant [23 x i8] c"_ZTSZ9host_onlyvEUlvE_\00", align 1
// CIR-LLVM-HOST: @[[INT:__usn_str\.[0-9]+]] = private constant [6 x i8] c"_ZTSi\00", align 1
// CIR-LLVM-HOST: @[[CONSTEXPR_STR:[^ ]+]] = private constant [6 x i8] c"_ZTSl\00", align 1
// CIR-LLVM-HOST: @[[TPL_INT:__usn_str\.[0-9]+]] = private constant [6 x i8] c"_ZTSi\00", align 1
// CIR-LLVM-HOST: @[[TPL_LAMBDA:__usn_str\.[0-9]+]] = private constant [23 x i8] c"_ZTSZ9host_onlyvEUlvE_\00", align 1

// OGCG-HOST: @[[GLOBAL_STR:[^ ]+]] = private unnamed_addr constant [6 x i8] c"_ZTSi\00", align 1
// OGCG-HOST: @global_name = global ptr @[[GLOBAL_STR]], align 8
// OGCG-HOST: @[[LAMBDA:__usn_str]] = private unnamed_addr constant [23 x i8] c"_ZTSZ9host_onlyvEUlvE_\00", align 1
// OGCG-HOST: @[[INT:__usn_str\.[0-9]+]] = private unnamed_addr constant [6 x i8] c"_ZTSi\00", align 1
// OGCG-HOST: @[[CONSTEXPR_STR:[^ ]+]] = private unnamed_addr constant [6 x i8] c"_ZTSl\00", align 1
// OGCG-HOST: @[[TPL_INT:__usn_str\.[0-9]+]] = private unnamed_addr constant [6 x i8] c"_ZTSi\00", align 1
// OGCG-HOST: @[[TPL_LAMBDA:__usn_str\.[0-9]+]] = private unnamed_addr constant [23 x i8] c"_ZTSZ9host_onlyvEUlvE_\00", align 1

// CIR-DEV: cir.global "private" constant cir_private dso_local target_address_space(1) @[[KERNEL_LAMBDA:__usn_str]] = #cir.const_array<"_ZTSZZ4mainENKUlvE_clEvEUlvE_" : !cir.array<!s8i x 29>, trailing_zeros> : !cir.array<!s8i x 30> align(1)
// CIR-DEV: cir.global "private" constant cir_private dso_local target_address_space(1) @[[KERNEL_CONSTEXPR:[^ ]+]] = #cir.const_array<"_ZTSc" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> align(1)

// CIR-LLVM-DEV: @[[KERNEL_LAMBDA:__usn_str]] = private addrspace(1) constant [30 x i8] c"_ZTSZZ4mainENKUlvE_clEvEUlvE_\00", align 1
// CIR-LLVM-DEV: @[[KERNEL_CONSTEXPR:[^ ]+]] = private addrspace(1) constant [6 x i8] c"_ZTSc\00", align 1

// OGCG-DEV: @[[KERNEL_LAMBDA:__usn_str]] = private unnamed_addr addrspace(1) constant [30 x i8] c"_ZTSZZ4mainENKUlvE_clEvEUlvE_\00", align 1
// OGCG-DEV: @[[KERNEL_CONSTEXPR:[^ ]+]] = private unnamed_addr addrspace(1) constant [6 x i8] c"_ZTSc\00", align 1

template <typename T>
void template_param() {
  use(__builtin_sycl_unique_stable_name(T));
}

void host_only() {
  auto l = [] {};
  use(__builtin_sycl_unique_stable_name(decltype(l)));
  template_param<int>();
  template_param<decltype(l)>();

  // The operand is unevaluated.
  int i = 0;
  use(__builtin_sycl_unique_stable_name(decltype(i++)));

  // A constexpr use is constant-folded to a string literal.
  constexpr const char *c = __builtin_sycl_unique_stable_name(long);
  use(c);
}

// CIR-HOST-LABEL: cir.func {{.*}} @_Z9host_onlyv()
// CIR-HOST:   %[[L:.*]] = cir.get_global @[[LAMBDA]] : !cir.ptr<!cir.array<!s8i x 23>>
// CIR-HOST:   %[[L_CAST:.*]] = cir.cast bitcast %[[L]] : !cir.ptr<!cir.array<!s8i x 23>> -> !cir.ptr<!s8i>
// CIR-HOST:   cir.call @_Z3usePKc(%[[L_CAST]])
// CIR-HOST:   %[[I:.*]] = cir.get_global @[[INT]] : !cir.ptr<!cir.array<!s8i x 6>>
// CIR-HOST:   %[[I_CAST:.*]] = cir.cast bitcast %[[I]] : !cir.ptr<!cir.array<!s8i x 6>> -> !cir.ptr<!s8i>
// CIR-HOST:   cir.call @_Z3usePKc(%[[I_CAST]])
// CIR-HOST:   cir.const #cir.global_view<@[[CONSTEXPR_STR]]> : !cir.ptr<!s8i>

// LLVM-HOST-LABEL: define dso_local void @_Z9host_onlyv()
// LLVM-HOST:   call void @_Z3usePKc(ptr noundef @[[LAMBDA]])
// LLVM-HOST:   call void @_Z3usePKc(ptr noundef @[[INT]])
// LLVM-HOST:   store ptr @[[CONSTEXPR_STR]], ptr %{{.*}}, align 8

// CIR-HOST-LABEL: cir.func {{.*}} @_Z14template_paramIiEvv()
// CIR-HOST:   %[[T:.*]] = cir.get_global @[[TPL_INT]] : !cir.ptr<!cir.array<!s8i x 6>>
// CIR-HOST:   %[[T_CAST:.*]] = cir.cast bitcast %[[T]] : !cir.ptr<!cir.array<!s8i x 6>> -> !cir.ptr<!s8i>
// CIR-HOST:   cir.call @_Z3usePKc(%[[T_CAST]])

// LLVM-HOST-LABEL: define linkonce_odr void @_Z14template_paramIiEvv()
// LLVM-HOST:   call void @_Z3usePKc(ptr noundef @[[TPL_INT]])

// CIR-HOST-LABEL: cir.func {{.*}} @_Z14template_paramIZ9host_onlyvEUlvE_Evv()
// CIR-HOST:   %[[T:.*]] = cir.get_global @[[TPL_LAMBDA]] : !cir.ptr<!cir.array<!s8i x 23>>
// CIR-HOST:   %[[T_CAST:.*]] = cir.cast bitcast %[[T]] : !cir.ptr<!cir.array<!s8i x 23>> -> !cir.ptr<!s8i>
// CIR-HOST:   cir.call @_Z3usePKc(%[[T_CAST]])

// LLVM-HOST-LABEL: define internal void @_Z14template_paramIZ9host_onlyvEUlvE_Evv()
// LLVM-HOST:   call void @_Z3usePKc(ptr noundef @[[TPL_LAMBDA]])

int main() {
  kernel_single_task<class K>([] {
    auto x = [] {};
    use(__builtin_sycl_unique_stable_name(decltype(x)));
    constexpr const char *c = __builtin_sycl_unique_stable_name(char);
    use(c);
  });
}

// On the device, the global is in the global constant address space and is
// cast to the generic address space of the expression type.

// CIR-DEV-LABEL: cir.func {{.*}} @_ZZ4mainENKUlvE_clEv(
// CIR-DEV:   %[[G:.*]] = cir.get_global @[[KERNEL_LAMBDA]] : !cir.ptr<!cir.array<!s8i x 30>, target_address_space(1)>
// CIR-DEV:   %[[AS:.*]] = cir.cast address_space %[[G]] : !cir.ptr<!cir.array<!s8i x 30>, target_address_space(1)> -> !cir.ptr<!cir.array<!s8i x 30>, target_address_space(4)>
// CIR-DEV:   %[[CAST:.*]] = cir.cast bitcast %[[AS]] : !cir.ptr<!cir.array<!s8i x 30>, target_address_space(4)> -> !cir.ptr<!s8i, target_address_space(4)>
// CIR-DEV:   cir.call @_Z3usePKc(%[[CAST]])
// CIR-DEV:   cir.const #cir.global_view<@[[KERNEL_CONSTEXPR]]> : !cir.ptr<!s8i, target_address_space(4)>

// LLVM-DEV-LABEL: define internal spir_func void @_ZZ4mainENKUlvE_clEv(
// LLVM-DEV:   call spir_func void @_Z3usePKc(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @[[KERNEL_LAMBDA]] to ptr addrspace(4)))
// LLVM-DEV:   store ptr addrspace(4) addrspacecast (ptr addrspace(1) @[[KERNEL_CONSTEXPR]] to ptr addrspace(4)), ptr addrspace(4) %{{.*}}, align 8

