// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=OGCG,LLVM-OGCG --input-file=%t.ll %s

// References to const, non-volatile variables whose initializer can be
// constant evaluated are emitted as the constant value, even when the variable
// is not usable in constant expressions (e.g. 'const float') and even when the
// reference is to an enclosing variable captured by a lambda.

// CIR-NOT: cir.global {{.*}}@__const.
// LLVM-OGCG-NOT: @__const.

float use(float);
int side();
template <class F> auto call(F f) { return f(); }

float captured() {
  const float k = 0.9f;
  const int n = 4;
  return call([=] { return use(k) + n; });
}

int non_captured() {
  const int n = 4;
  return n;
}

// CIR-LABEL: cir.func {{.*}} @_Z12non_capturedv()
// CIR:         %[[RV:.*]] = cir.alloca "__retval"
// CIR:         %[[N:.*]] = cir.alloca "n"
// CIR:         %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CIR:         cir.store {{.*}}%[[FOUR]], %[[N]]
// CIR:         %[[FOUR2:.*]] = cir.const #cir.int<4> : !s32i
// CIR:         cir.store {{.*}}%[[FOUR2]], %[[RV]]

// LLVM-OGCG-LABEL: define {{.*}}i32 @_Z12non_capturedv()
// LLVM-OGCG:         store i32 4, ptr %{{.*}}
// LLVM:              store i32 4, ptr %{{.*}}
// OGCG:              ret i32 4

int constexpr_var() {
  constexpr int c = 7;
  return c * 2;
}

// CIR-LABEL: cir.func {{.*}} @_Z13constexpr_varv()
// CIR:         %[[C:.*]] = cir.alloca "c"
// CIR:         cir.store {{.*}}, %[[C]]
// CIR:         %[[SEVEN:.*]] = cir.const #cir.int<7> : !s32i
// CIR:         %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:         cir.mul nsw %[[SEVEN]], %[[TWO]] : !s32i

// LLVM-OGCG-LABEL: define {{.*}}i32 @_Z13constexpr_varv()
// LLVM:              store i32 14, ptr %{{.*}}
// OGCG:              ret i32 14

_Complex float complex_var() {
  const _Complex float z = 1.0f;
  return z;
}

// CIR-LABEL: cir.func {{.*}} @_Z11complex_varv()
// CIR:         %[[RV:.*]] = cir.alloca "__retval"
// CIR:         %[[Z_ADDR:.*]] = cir.alloca "z"
// CIR:         cir.store {{.*}}, %[[Z_ADDR]]
// CIR:         %[[Z:.*]] = cir.const #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<0.000000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR:         cir.store {{.*}}%[[Z]], %[[RV]]

// LLVM-OGCG-LABEL: define {{.*}} @_Z11complex_varv()
// LLVM:              store { float, float } { float 1.000000e+00, float 0.000000e+00 }, ptr %{{.*}}
// OGCG:              store float 1.000000e+00, ptr %{{.*}}
// OGCG:              store float 0.000000e+00, ptr %{{.*}}

int global_var = 5;
int captured_ref_to_global() {
  int &r = global_var;
  return call([=] { return r; });
}

// References whose initializer is a constant l-value are emitted as a load of
// the referenced object.

int const_ref_to_nonconst_global() {
  const int &r = global_var;
  return r;
}

// CIR-LABEL: cir.func {{.*}} @_Z28const_ref_to_nonconst_globalv()
// CIR:         %[[R:.*]] = cir.alloca "r"
// CIR:         cir.store {{.*}}, %[[R]]
// CIR:         %[[G:.*]] = cir.const #cir.global_view<@global_var> : !cir.ptr<!s32i>
// CIR:         cir.load {{.*}}%[[G]] : !cir.ptr<!s32i>, !s32i

// LLVM-OGCG-LABEL: define {{.*}}i32 @_Z28const_ref_to_nonconst_globalv()
// LLVM-OGCG:         store ptr @global_var, ptr %{{.*}}
// LLVM-OGCG:         load i32, ptr @global_var

int g_arr[4] = {1, 2, 3, 4};
int ref_to_array_element() {
  int &r = g_arr[1];
  return r;
}

// CIR-LABEL: cir.func {{.*}} @_Z20ref_to_array_elementv()
// CIR:         %[[R:.*]] = cir.alloca "r"
// CIR:         cir.store {{.*}}, %[[R]]
// CIR:         %[[ELT:.*]] = cir.const #cir.global_view<@g_arr, [1 : i32]> : !cir.ptr<!s32i>
// CIR:         cir.load {{.*}}%[[ELT]] : !cir.ptr<!s32i>, !s32i

// LLVM-OGCG-LABEL: define {{.*}}i32 @_Z20ref_to_array_elementv()
// LLVM-OGCG:         load i32, ptr getelementptr {{.*}}(i8, ptr @g_arr, i64 4)

_Complex float gcf;
float complex_ref() {
  _Complex float &r = gcf;
  return __imag__ (r + r);
}

// CIR-LABEL: cir.func {{.*}} @_Z11complex_refv()
// CIR:         %[[R:.*]] = cir.alloca "r"
// CIR:         cir.store {{.*}}, %[[R]]
// CIR:         %[[G0:.*]] = cir.const #cir.global_view<@gcf> : !cir.ptr<!cir.complex<!cir.float>>
// CIR:         %[[V0:.*]] = cir.load {{.*}}%[[G0]]
// CIR:         %[[G1:.*]] = cir.const #cir.global_view<@gcf> : !cir.ptr<!cir.complex<!cir.float>>
// CIR:         %[[V1:.*]] = cir.load {{.*}}%[[G1]]
// CIR:         cir.complex.fadd %[[V0]], %[[V1]]

// LLVM-OGCG-LABEL: define {{.*}}float @_Z11complex_refv()
// LLVM:              load { float, float }, ptr @gcf
// LLVM:              load { float, float }, ptr @gcf
// OGCG:              load float, ptr @gcf
// OGCG:              load float, ptr getelementptr inbounds nuw (i8, ptr @gcf, i64 4)

// A const pointer whose value is the address of a global.

int const_ptr_to_global() {
  int *const p = &global_var;
  return *p;
}

// CIR-LABEL: cir.func {{.*}} @_Z19const_ptr_to_globalv()
// CIR:         %[[P:.*]] = cir.alloca "p"
// CIR:         cir.store {{.*}}, %[[P]]
// CIR:         %[[G:.*]] = cir.const #cir.global_view<@global_var> : !cir.ptr<!s32i>
// CIR:         cir.load {{.*}}%[[G]] : !cir.ptr<!s32i>, !s32i

// LLVM-OGCG-LABEL: define {{.*}}i32 @_Z19const_ptr_to_globalv()
// LLVM-OGCG:         load i32, ptr @global_var

// Constant pointers and integers that need dedicated LLVM constant expressions.

int const_absolute_ptr(void *p) {
  void *const failed = (void *)-1;
  return p == failed;
}

// CIR-LABEL: cir.func {{.*}} @_Z18const_absolute_ptrPv(
// CIR:         %[[FAILED:.*]] = cir.const #cir.ptr<-1 : i64> : !cir.ptr<!void>
// CIR:         cir.cmp eq %{{.*}}, %[[FAILED]] : !cir.ptr<!void>

// LLVM-OGCG-LABEL: define {{.*}}i32 @_Z18const_absolute_ptrPv(
// LLVM-OGCG:         icmp eq ptr %{{.*}}, inttoptr (i64 -1 to ptr)

void *const_label_addr(int i) {
  void *const target = &&l1;
  const long diff = (long)&&l2 - (long)&&l1;
  if (i)
    goto *target;
  return (void *)diff;
l1:
  return 0;
l2:
  return (void *)1;
}

// CIR-LABEL: cir.func {{.*}} @_Z16const_label_addri(
// CIR:         %[[TARGET:.*]] = cir.const #cir.block_addr_info<@_Z16const_label_addri, "l1"> : !cir.ptr<!void>
// CIR:         cir.indirect_goto %[[TARGET]] : !cir.ptr<!void>
// CIR:         %[[DIFF:.*]] = cir.const #cir.block_addr_diff<@_Z16const_label_addri, "l2", "l1"> : !s64i
// CIR:         cir.cast integral %[[DIFF]] : !s64i -> !u64i

// LLVM-OGCG-LABEL: define {{.*}}ptr @_Z16const_label_addri(
// LLVM-OGCG:         store ptr inttoptr (i64 sub (i64 ptrtoint (ptr blockaddress(@_Z16const_label_addri, %[[L2:.*]]) to i64), i64 ptrtoint (ptr blockaddress(@_Z16const_label_addri, %[[L1:.*]]) to i64)) to ptr), ptr %{{.*}}
// LLVM-OGCG:         phi ptr [ blockaddress(@_Z16const_label_addri, %[[L1]]), %{{.*}} ]

// Static data members named through a member access.

struct S {
  static constexpr float kf = 1.5f;
};
float static_member(S s) { return s.kf; }

// CIR-LABEL: cir.func {{.*}} @_Z13static_member1S(
// CIR-NOT:     cir.get_global
// CIR:         cir.const #cir.fp<1.500000e+00> : !cir.float

// LLVM-OGCG-LABEL: define {{.*}}float @_Z13static_member1S(
// LLVM-OGCG-NOT:     load float
// LLVM:              store float 1.500000e+00
// OGCG:              ret float 1.500000e+00

// Negative cases: the value is loaded from the variable.

int volatile_const() {
  volatile const int v = 3;
  return v;
}

// CIR-LABEL: cir.func {{.*}} @_Z14volatile_constv()
// CIR:         %[[V:.*]] = cir.alloca "v"
// CIR:         cir.load volatile {{.*}}%[[V]]

// LLVM-OGCG-LABEL: define {{.*}}i32 @_Z14volatile_constv()
// LLVM-OGCG:         load volatile i32, ptr %{{.*}}

int non_const_init() {
  const int x = side();
  return x;
}

// CIR-LABEL: cir.func {{.*}} @_Z14non_const_initv()
// CIR:         %[[X:.*]] = cir.alloca "x"
// CIR:         cir.call @_Z4sidev()
// CIR:         cir.load {{.*}}%[[X]]

// LLVM-OGCG-LABEL: define {{.*}}i32 @_Z14non_const_initv()
// LLVM-OGCG:         %[[X:.*]] = alloca i32
// LLVM-OGCG:         call {{.*}}i32 @_Z4sidev()
// LLVM-OGCG:         load i32, ptr %[[X]]

int param(const int p) {
  return p;
}

// CIR-LABEL: cir.func {{.*}} @_Z5parami(
// CIR:         %[[P:.*]] = cir.alloca "p"
// CIR:         cir.load {{.*}}%[[P]]

// LLVM-OGCG-LABEL: define {{.*}}i32 @_Z5parami(
// LLVM-OGCG:         %[[P:.*]] = alloca i32
// LLVM-OGCG:         load i32, ptr %[[P]]

// The captured constants are used directly instead of being loaded from the
// closure.

// CIR-LABEL: cir.func {{.*}} lambda {{.*}} @_ZZ8capturedvENK3$_0clEv(
// CIR-NOT:     cir.get_member
// CIR:         %[[K:.*]] = cir.const #cir.fp<0.899999976> : !cir.float
// CIR:         %[[USE:.*]] = cir.call @_Z3usef(%[[K]])
// CIR:         %[[N:.*]] = cir.const #cir.int<4> : !s32i
// CIR:         %[[NF:.*]] = cir.cast int_to_float %[[N]] : !s32i -> !cir.float
// CIR:         cir.fadd %[[USE]], %[[NF]] : !cir.float

// LLVM-OGCG-LABEL: define internal {{.*}}float @"_ZZ8capturedvENK3$_0clEv"(
// LLVM-OGCG-NOT:     getelementptr
// LLVM-OGCG:         %[[USE:.*]] = call {{.*}}float @_Z3usef(float noundef f0x3F666666)
// LLVM-OGCG:         fadd float %[[USE]], 4.000000e+00

// A reference to a global is emitted as a reference constant: the global is
// loaded directly rather than through the reference.

// CIR-LABEL: cir.func {{.*}} lambda {{.*}} @_ZZ22captured_ref_to_globalvENK3$_0clEv(
// CIR:         %[[G:.*]] = cir.const #cir.global_view<@global_var> : !cir.ptr<!s32i>
// CIR:         cir.load {{.*}}%[[G]] : !cir.ptr<!s32i>, !s32i

// LLVM-OGCG-LABEL: define internal {{.*}}i32 @"_ZZ22captured_ref_to_globalvENK3$_0clEv"(
// LLVM-OGCG:         load i32, ptr @global_var
