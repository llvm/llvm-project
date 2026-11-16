// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

// Adapted from CodeGenCXX/vla-lambda-capturing.cpp.

typedef __INTPTR_TYPE__ intptr_t;

// CIR: ![[CAP_TYPE1:.+]] = !cir.struct<class "anon.0" {data !u64i, data !cir.ptr<!s64i>, data !cir.ptr<!s64i>}>
// CIR: ![[CAP_TYPE2:.+]] = !cir.struct<class "anon.1" {data !u64i, data !cir.ptr<!s64i>}>
// CIR: ![[CAP_TYPE3:.+]] = !cir.struct<class "anon.2" {data !cir.ptr<!s64i>, data !u64i, data !u64i, data !cir.ptr<!s64i>, data !cir.ptr<!s64i>}>
// CIR: ![[CAP_TYPE4:.+]] = !cir.struct<class "anon.3" {data !cir.ptr<!s64i>, data !u64i, data !cir.ptr<!s64i>, data !u64i, data !cir.ptr<!s64i>}>

// LLVM-DAG: %[[CAP_TYPE1:.+]] = type { i64, ptr, ptr }
// LLVM-DAG: %[[CAP_TYPE2:.+]] = type { i64, ptr }
// LLVM-DAG: %[[CAP_TYPE3:.+]] = type { ptr, i64, i64, ptr, ptr }
// LLVM-DAG: %[[CAP_TYPE4:.+]] = type { ptr, i64, ptr, i64, ptr }

void g(intptr_t n) {
  intptr_t buffer[n];
  [&buffer, &n]() {
    __typeof(buffer) x;
  }();
}

// CIR: cir.func {{.*}} @[[G:.+]](%
// CIR: %[[N_TEMP:.+]] = cir.alloca "n" align(8) init : !cir.ptr<!s64i>
// CIR: %[[LAMBDA_TEMP:.+]] = cir.alloca "ref.tmp0" align(8) : !cir.ptr<![[CAP_TYPE1]]>
// CIR: %[[N_LOAD:.+]] = cir.load align(8) %[[N_TEMP]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[N_CAST:.+]] = cir.cast integral %[[N_LOAD]] : !s64i -> !u64i
// CIR: cir.stacksave
// CIR: %[[BUF_TEMP:.+]] = cir.alloca "buffer" {{.*}} size(%[[N_CAST]])
// CIR: %[[VLA_SIZE:.+]] = cir.get_member %[[LAMBDA_TEMP]][0] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE1]]> -> !cir.ptr<!u64i>
// CIR: cir.store align(8) %[[N_CAST]], %[[VLA_SIZE]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[BUF_CAP:.+]] = cir.get_member %[[LAMBDA_TEMP]][1] {name = "buffer"} : !cir.ptr<![[CAP_TYPE1]]> -> !cir.ptr<!cir.ptr<!s64i>>
// CIR: cir.store align(8) %[[BUF_TEMP]], %[[BUF_CAP]] : !cir.ptr<!s64i>, !cir.ptr<!cir.ptr<!s64i>>
// CIR: %[[N_CAP:.+]] = cir.get_member %[[LAMBDA_TEMP]][2] {name = "n"} : !cir.ptr<![[CAP_TYPE1]]> -> !cir.ptr<!cir.ptr<!s64i>>
// CIR: cir.store align(8) %[[N_TEMP]], %[[N_CAP]] : !cir.ptr<!s64i>, !cir.ptr<!cir.ptr<!s64i>>
// CIR: cir.call @[[G_LAMBDA:.+]](%[[LAMBDA_TEMP]]) : (!cir.ptr<![[CAP_TYPE1]]> {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) -> ()
// CIR: cir.stackrestore

// CIR: cir.func {{.*}} @[[G_LAMBDA]]
// CIR: %[[THIS:.+]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<![[CAP_TYPE1]]>>
// CIR: %[[THIS_LOAD:.+]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<![[CAP_TYPE1]]>>, !cir.ptr<![[CAP_TYPE1]]>
// CIR: %[[L_VLA_SIZE:.+]] = cir.get_member %[[THIS_LOAD]][0] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE1]]> -> !cir.ptr<!u64i>
// CIR: %[[L_N:.+]] = cir.load align(8) %[[L_VLA_SIZE]] : !cir.ptr<!u64i>, !u64i
// CIR: cir.stacksave
// CIR: cir.alloca "x" {{.*}} size(%[[L_N]])
// CIR: cir.stackrestore

// LLVM: define {{.*}} void @[[G:.+]](
// LLVM: %[[N_TEMP:.+]] = alloca i64, align 8
// LLVM: %[[LAMBDA_TEMP:.+]] = alloca %[[CAP_TYPE1]], align 8
// LLVM: %[[N_LOAD:.+]] = load i64, ptr %[[N_TEMP]], align 8
// LLVM: call ptr @llvm.stacksave.p0()
// LLVM: %[[BUF_TEMP:.+]] = alloca i64, i64 %[[N_LOAD]], align 16
// LLVM: %[[VLA_SIZE:.+]] = getelementptr inbounds nuw %[[CAP_TYPE1]], ptr %[[LAMBDA_TEMP]], i32 0, i32 0
// LLVM: store i64 %[[N_LOAD]], ptr %[[VLA_SIZE]], align 8
// LLVM: %[[BUF_CAP:.+]] = getelementptr inbounds nuw %[[CAP_TYPE1]], ptr %[[LAMBDA_TEMP]], i32 0, i32 1
// LLVM: store ptr %[[BUF_TEMP]], ptr %[[BUF_CAP]], align 8
// LLVM: %[[N_CAP:.+]] = getelementptr inbounds nuw %[[CAP_TYPE1]], ptr %[[LAMBDA_TEMP]], i32 0, i32 2
// LLVM: store ptr %[[N_TEMP]], ptr %[[N_CAP]], align 8
// LLVM: call void @[[G_LAMBDA:.+]](ptr {{.*}}%[[LAMBDA_TEMP]])
// LLVM: call void @llvm.stackrestore.p0(

// LLVM: define {{.*}} void @[[G_LAMBDA]](
// LLVM: %[[THIS:.+]] = alloca ptr, align 8
// LLVM: %[[THIS_LOAD:.+]] = load ptr, ptr %[[THIS]], align 8
// LLVM: %[[L_VLA_SIZE:.+]] = getelementptr inbounds nuw %[[CAP_TYPE1]], ptr %[[THIS_LOAD]], i32 0, i32 0
// LLVM: %[[L_N:.+]] = load i64, ptr %[[L_VLA_SIZE]], align 8
// LLVM: call ptr @llvm.stacksave.p0()
// LLVM: alloca i64, i64 %[[L_N]], align 16
// LLVM: call void @llvm.stackrestore.p0(

template <typename T>
void f(T n, T m) {
  intptr_t buffer[n + m];
  [&buffer]() {
    __typeof(buffer) x;
  }();
}

template <typename T>
intptr_t getSize(T);

template <typename T>
void b(intptr_t n, T arg) {
  typedef intptr_t ArrTy[getSize(arg)];
  ArrTy buffer2;
  ArrTy buffer1[n + arg];
  intptr_t a;
  [&]() {
    n = sizeof(buffer1[n]);
    [&]() {
      n = sizeof(buffer2);
      n = sizeof(buffer1);
    }();
  }();
}
// CIR-LABEL: @main
// LLVM-LABEL: @main
int main() {
  // CIR:  cir.call @[[G]](
  // LLVM:  call {{.*}}void @[[G]](
  g((intptr_t)1);
  // CIR:  cir.call @[[F_INT:.+]](%
  // LLVM:  call {{.*}}void @[[F_INT:.+]](
  f((intptr_t)1, (intptr_t)2);
  // CIR:  cir.call @[[B_INT:.+]](%
  // LLVM:  call {{.*}}void @[[B_INT:.+]](
  b((intptr_t)12, (intptr_t)13);
  return 0;
}


// CIR: cir.func {{.*}} @[[F_INT]]
// CIR: %[[F_N_TEMP:.+]] = cir.alloca "n" align(8) init : !cir.ptr<!s64i>
// CIR: %[[F_M_TEMP:.+]] = cir.alloca "m" align(8) init : !cir.ptr<!s64i>
// CIR: %[[F_LAMBDA_TEMP:.+]] = cir.alloca "ref.tmp0" align(8) : !cir.ptr<![[CAP_TYPE2]]>
// CIR: %[[F_N_LOAD:.+]] = cir.load align(8) %[[F_N_TEMP]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[F_M_LOAD:.+]] = cir.load align(8) %[[F_M_TEMP]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[F_SUM:.+]] = cir.add nsw %[[F_N_LOAD]], %[[F_M_LOAD]] : !s64i
// CIR: %[[F_CAST:.+]] = cir.cast integral %[[F_SUM]] : !s64i -> !u64i
// CIR: cir.stacksave
// CIR: %[[F_BUF_TEMP:.+]] = cir.alloca "buffer" {{.*}} size(%[[F_CAST]])
// CIR: %[[F_VLA_SIZE:.+]] = cir.get_member %[[F_LAMBDA_TEMP]][0] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE2]]> -> !cir.ptr<!u64i>
// CIR: cir.store align(8) %[[F_CAST]], %[[F_VLA_SIZE]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[F_BUF_CAP:.+]] = cir.get_member %[[F_LAMBDA_TEMP]][1] {name = "buffer"} : !cir.ptr<![[CAP_TYPE2]]> -> !cir.ptr<!cir.ptr<!s64i>>
// CIR: cir.store align(8) %[[F_BUF_TEMP]], %[[F_BUF_CAP]] : !cir.ptr<!s64i>, !cir.ptr<!cir.ptr<!s64i>>
// CIR: cir.call [[F_LAMBDA:@.+]](%[[F_LAMBDA_TEMP]]) : (!cir.ptr<![[CAP_TYPE2]]> {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) -> ()
// CIR: cir.stackrestore

// CIR: cir.func {{.*}} @[[B_INT]]
// CIR: %[[B_N_TEMP:.+]] = cir.alloca "n" align(8) init : !cir.ptr<!s64i>
// CIR: %[[B_ARG_TEMP:.+]] = cir.alloca "arg" align(8) init : !cir.ptr<!s64i>
// CIR: %[[B_LAMBDA_TEMP:.+]] = cir.alloca "ref.tmp0" align(8) : !cir.ptr<![[CAP_TYPE3]]>
// CIR: %[[B_ARG_LOAD:.+]] = cir.load align(8) %[[B_ARG_TEMP]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[B_GET_SIZE:.+]] = cir.call @_Z7getSizeIlElT_(%[[B_ARG_LOAD]]) : (!s64i {llvm.noundef}) -> (!s64i {llvm.noundef})
// CIR: %[[B_SIZE1:.+]] = cir.cast integral %[[B_GET_SIZE]] : !s64i -> !u64i
// CIR: cir.stacksave
// CIR: %[[B_BUF2_TEMP:.+]] = cir.alloca "buffer2" {{.*}} size(%[[B_SIZE1]])
// CIR: %[[B_N_LOAD:.+]] = cir.load align(8) %[[B_N_TEMP]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[B_ARG_LOAD2:.+]] = cir.load align(8) %[[B_ARG_TEMP]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[B_SUM:.+]] = cir.add nsw %[[B_N_LOAD]], %[[B_ARG_LOAD2]] : !s64i
// CIR: %[[B_SIZE2:.+]] = cir.cast integral %[[B_SUM]] : !s64i -> !u64i
// CIR: %[[B_TOTAL:.+]] = cir.mul nuw %[[B_SIZE2]], %[[B_SIZE1]] : !u64i
// CIR: %[[B_BUF1_TEMP:.+]] = cir.alloca "buffer1" {{.*}} size(%[[B_TOTAL]])
// CIR: %[[B_N_CAP:.+]] = cir.get_member %[[B_LAMBDA_TEMP]][0] {name = "n"} : !cir.ptr<![[CAP_TYPE3]]> -> !cir.ptr<!cir.ptr<!s64i>>
// CIR: cir.store align(8) %[[B_N_TEMP]], %[[B_N_CAP]] : !cir.ptr<!s64i>, !cir.ptr<!cir.ptr<!s64i>>
// CIR: %[[B_VLA_SIZE2:.+]] = cir.get_member %[[B_LAMBDA_TEMP]][1] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE3]]> -> !cir.ptr<!u64i>
// CIR: cir.store align(8) %[[B_SIZE2]], %[[B_VLA_SIZE2]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[B_VLA_SIZE1:.+]] = cir.get_member %[[B_LAMBDA_TEMP]][2] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE3]]> -> !cir.ptr<!u64i>
// CIR: cir.store align(8) %[[B_SIZE1]], %[[B_VLA_SIZE1]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[B_BUF1_CAP:.+]] = cir.get_member %[[B_LAMBDA_TEMP]][3] {name = "buffer1"} : !cir.ptr<![[CAP_TYPE3]]> -> !cir.ptr<!cir.ptr<!s64i>>
// CIR: cir.store align(8) %[[B_BUF1_TEMP]], %[[B_BUF1_CAP]] : !cir.ptr<!s64i>, !cir.ptr<!cir.ptr<!s64i>>
// CIR: %[[B_BUF2_CAP:.+]] = cir.get_member %[[B_LAMBDA_TEMP]][4] {name = "buffer2"} : !cir.ptr<![[CAP_TYPE3]]> -> !cir.ptr<!cir.ptr<!s64i>>
// CIR: cir.store align(8) %[[B_BUF2_TEMP]], %[[B_BUF2_CAP]] : !cir.ptr<!s64i>, !cir.ptr<!cir.ptr<!s64i>>
// CIR: cir.call [[B_LAMBDA:@[^(]+]](%[[B_LAMBDA_TEMP]]) : (!cir.ptr<![[CAP_TYPE3]]> {llvm.align = 8 : i64, llvm.dereferenceable = 40 : i64, llvm.nonnull, llvm.noundef}) -> ()
// CIR: cir.stackrestore

// CIR: cir.func {{.*}} [[F_LAMBDA]]
// CIR: %[[F_THIS:.+]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<![[CAP_TYPE2]]>>
// CIR: %[[F_THIS_LOAD:.+]] = cir.load %[[F_THIS]] : !cir.ptr<!cir.ptr<![[CAP_TYPE2]]>>, !cir.ptr<![[CAP_TYPE2]]>
// CIR: %[[F_L_VLA_SIZE:.+]] = cir.get_member %[[F_THIS_LOAD]][0] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE2]]> -> !cir.ptr<!u64i>
// CIR: %[[F_L_N:.+]] = cir.load align(8) %[[F_L_VLA_SIZE]] : !cir.ptr<!u64i>, !u64i
// CIR: cir.stacksave
// CIR: cir.alloca "x" {{.*}} size(%[[F_L_N]])
// CIR: cir.stackrestore

// CIR: cir.func {{.*}} [[B_LAMBDA]]
// CIR: %[[BL_THIS:.+]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<![[CAP_TYPE3]]>>
// CIR: %[[BL_INNER_TEMP:.+]] = cir.alloca "ref.tmp0" align(8) : !cir.ptr<![[CAP_TYPE4]]>
// CIR: %[[BL_THIS_LOAD:.+]] = cir.load %[[BL_THIS]] : !cir.ptr<!cir.ptr<![[CAP_TYPE3]]>>, !cir.ptr<![[CAP_TYPE3]]>
// CIR: %[[BL_VLA_SIZE2:.+]] = cir.get_member %[[BL_THIS_LOAD]][1] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE3]]> -> !cir.ptr<!u64i>
// CIR: %[[BL_SIZE2:.+]] = cir.load align(8) %[[BL_VLA_SIZE2]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[BL_VLA_SIZE1:.+]] = cir.get_member %[[BL_THIS_LOAD]][2] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE3]]> -> !cir.ptr<!u64i>
// CIR: %[[BL_SIZE1:.+]] = cir.load align(8) %[[BL_VLA_SIZE1]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[BL_BUF1_CAP:.+]] = cir.get_member %[[BL_THIS_LOAD]][3] {name = "buffer1"} : !cir.ptr<![[CAP_TYPE3]]> -> !cir.ptr<!cir.ptr<!s64i>>
// CIR: %[[BL_BUF1:.+]] = cir.load %[[BL_BUF1_CAP]] : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// CIR: %[[BL_OFFSET:.+]] = cir.mul nsw %{{.+}}, %[[BL_SIZE1]] : !u64i
// CIR: cir.ptr_stride %[[BL_BUF1]], %[[BL_OFFSET]] : (!cir.ptr<!s64i>, !u64i) -> !cir.ptr<!s64i>
// CIR: cir.mul nuw %{{.+}}, %[[BL_SIZE1]] : !u64i
// CIR: %[[BL_INNER_VLA_SIZE1:.+]] = cir.get_member %[[BL_INNER_TEMP]][1] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE4]]> -> !cir.ptr<!u64i>
// CIR: cir.store align(8) %[[BL_SIZE1]], %[[BL_INNER_VLA_SIZE1]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[BL_INNER_VLA_SIZE2:.+]] = cir.get_member %[[BL_INNER_TEMP]][3] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE4]]> -> !cir.ptr<!u64i>
// CIR: cir.store align(8) %[[BL_SIZE2]], %[[BL_INNER_VLA_SIZE2]] : !u64i, !cir.ptr<!u64i>
// CIR: cir.call [[B_LAMBDA_LAMBDA:@[^(]+]](%[[BL_INNER_TEMP]]) : (!cir.ptr<![[CAP_TYPE4]]> {llvm.align = 8 : i64, llvm.dereferenceable = 40 : i64, llvm.nonnull, llvm.noundef}) -> ()

// CIR: cir.func {{.*}} [[B_LAMBDA_LAMBDA]]
// CIR: %[[BLL_THIS:.+]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<![[CAP_TYPE4]]>>
// CIR: %[[BLL_THIS_LOAD:.+]] = cir.load %[[BLL_THIS]] : !cir.ptr<!cir.ptr<![[CAP_TYPE4]]>>, !cir.ptr<![[CAP_TYPE4]]>
// CIR: %[[BLL_VLA_SIZE1:.+]] = cir.get_member %[[BLL_THIS_LOAD]][1] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE4]]> -> !cir.ptr<!u64i>
// CIR: %[[BLL_SIZE1:.+]] = cir.load align(8) %[[BLL_VLA_SIZE1]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[BLL_VLA_SIZE2:.+]] = cir.get_member %[[BLL_THIS_LOAD]][3] {name = "vla-size"} : !cir.ptr<![[CAP_TYPE4]]> -> !cir.ptr<!u64i>
// CIR: %[[BLL_SIZE2:.+]] = cir.load align(8) %[[BLL_VLA_SIZE2]] : !cir.ptr<!u64i>, !u64i
// CIR: cir.mul nuw %{{.+}}, %[[BLL_SIZE1]] : !u64i
// CIR: %[[BLL_TOTAL:.+]] = cir.mul nuw %[[BLL_SIZE2]], %[[BLL_SIZE1]] : !u64i
// CIR: cir.mul nuw %{{.+}}, %[[BLL_TOTAL]] : !u64i


// LLVM: define {{.*}} void @[[F_INT]](
// LLVM: %[[F_N_TEMP:.+]] = alloca i64, align 8
// LLVM: %[[F_M_TEMP:.+]] = alloca i64, align 8
// LLVM: %[[F_LAMBDA_TEMP:.+]] = alloca %[[CAP_TYPE2]], align 8
// LLVM: %[[F_N_LOAD:.+]] = load i64, ptr %[[F_N_TEMP]], align 8
// LLVM: %[[F_M_LOAD:.+]] = load i64, ptr %[[F_M_TEMP]], align 8
// LLVM: %[[F_SUM:.+]] = add nsw i64 %[[F_N_LOAD]], %[[F_M_LOAD]]
// LLVM: call ptr @llvm.stacksave.p0()
// LLVM: %[[F_BUF_TEMP:.+]] = alloca i64, i64 %[[F_SUM]], align 16
// LLVM: %[[F_VLA_SIZE:.+]] = getelementptr inbounds nuw %[[CAP_TYPE2]], ptr %[[F_LAMBDA_TEMP]], i32 0, i32 0
// LLVM: store i64 %[[F_SUM]], ptr %[[F_VLA_SIZE]], align 8
// LLVM: %[[F_BUF_CAP:.+]] = getelementptr inbounds nuw %[[CAP_TYPE2]], ptr %[[F_LAMBDA_TEMP]], i32 0, i32 1
// LLVM: store ptr %[[F_BUF_TEMP]], ptr %[[F_BUF_CAP]], align 8
// LLVM: call void @[[F_LAMBDA:.+]](ptr {{.*}}%[[F_LAMBDA_TEMP]])
// LLVM: call void @llvm.stackrestore.p0(

// LLVM: define {{.*}} void @[[B_INT]](
// LLVM: %[[B_N_TEMP:.+]] = alloca i64, align 8
// LLVM: %[[B_ARG_TEMP:.+]] = alloca i64, align 8
// LLVM: %[[B_LAMBDA_TEMP:.+]] = alloca %[[CAP_TYPE3]], align 8
// LLVM: %[[B_ARG_LOAD:.+]] = load i64, ptr %[[B_ARG_TEMP]], align 8
// LLVM: %[[B_SIZE1:.+]] = call {{.*}}i64 @_Z7getSizeIlElT_(i64 {{.*}}%[[B_ARG_LOAD]])
// LLVM: call ptr @llvm.stacksave.p0()
// LLVM: %[[B_BUF2_TEMP:.+]] = alloca i64, i64 %[[B_SIZE1]], align 16
// LLVM: %[[B_N_LOAD:.+]] = load i64, ptr %[[B_N_TEMP]], align 8
// LLVM: %[[B_ARG_LOAD2:.+]] = load i64, ptr %[[B_ARG_TEMP]], align 8
// LLVM: %[[B_SIZE2:.+]] = add nsw i64 %[[B_N_LOAD]], %[[B_ARG_LOAD2]]
// LLVM: %[[B_TOTAL:.+]] = mul nuw i64 %[[B_SIZE2]], %[[B_SIZE1]]
// LLVM: %[[B_BUF1_TEMP:.+]] = alloca i64, i64 %[[B_TOTAL]], align 16
// LLVM: %[[B_N_CAP:.+]] = getelementptr inbounds nuw %[[CAP_TYPE3]], ptr %[[B_LAMBDA_TEMP]], i32 0, i32 0
// LLVM: store ptr %[[B_N_TEMP]], ptr %[[B_N_CAP]], align 8
// LLVM: %[[B_VLA_SIZE2:.+]] = getelementptr inbounds nuw %[[CAP_TYPE3]], ptr %[[B_LAMBDA_TEMP]], i32 0, i32 1
// LLVM: store i64 %[[B_SIZE2]], ptr %[[B_VLA_SIZE2]], align 8
// LLVM: %[[B_VLA_SIZE1:.+]] = getelementptr inbounds nuw %[[CAP_TYPE3]], ptr %[[B_LAMBDA_TEMP]], i32 0, i32 2
// LLVM: store i64 %[[B_SIZE1]], ptr %[[B_VLA_SIZE1]], align 8
// LLVM: %[[B_BUF1_CAP:.+]] = getelementptr inbounds nuw %[[CAP_TYPE3]], ptr %[[B_LAMBDA_TEMP]], i32 0, i32 3
// LLVM: store ptr %[[B_BUF1_TEMP]], ptr %[[B_BUF1_CAP]], align 8
// LLVM: %[[B_BUF2_CAP:.+]] = getelementptr inbounds nuw %[[CAP_TYPE3]], ptr %[[B_LAMBDA_TEMP]], i32 0, i32 4
// LLVM: store ptr %[[B_BUF2_TEMP]], ptr %[[B_BUF2_CAP]], align 8
// LLVM: call void @[[B_LAMBDA:.+]](ptr {{.*}}%[[B_LAMBDA_TEMP]])
// LLVM: call void @llvm.stackrestore.p0(

// LLVM: define {{.*}} void @[[F_LAMBDA]](
// LLVM: %[[F_THIS:.+]] = alloca ptr, align 8
// LLVM: %[[F_THIS_LOAD:.+]] = load ptr, ptr %[[F_THIS]], align 8
// LLVM: %[[F_L_VLA_SIZE:.+]] = getelementptr inbounds nuw %[[CAP_TYPE2]], ptr %[[F_THIS_LOAD]], i32 0, i32 0
// LLVM: %[[F_L_N:.+]] = load i64, ptr %[[F_L_VLA_SIZE]], align 8
// LLVM: call ptr @llvm.stacksave.p0()
// LLVM: alloca i64, i64 %[[F_L_N]], align 16
// LLVM: call void @llvm.stackrestore.p0(

// LLVM: define {{.*}} void @[[B_LAMBDA]](
// LLVM: %[[BL_THIS:.+]] = alloca ptr, align 8
// LLVM: %[[BL_INNER_TEMP:.+]] = alloca %[[CAP_TYPE4]], align 8
// LLVM: %[[BL_THIS_LOAD:.+]] = load ptr, ptr %[[BL_THIS]], align 8
// LLVM: %[[BL_VLA_SIZE2:.+]] = getelementptr inbounds nuw %[[CAP_TYPE3]], ptr %[[BL_THIS_LOAD]], i32 0, i32 1
// LLVM: %[[BL_SIZE2:.+]] = load i64, ptr %[[BL_VLA_SIZE2]], align 8
// LLVM: %[[BL_VLA_SIZE1:.+]] = getelementptr inbounds nuw %[[CAP_TYPE3]], ptr %[[BL_THIS_LOAD]], i32 0, i32 2
// LLVM: %[[BL_SIZE1:.+]] = load i64, ptr %[[BL_VLA_SIZE1]], align 8
// LLVM: %[[BL_BUF1_CAP:.+]] = getelementptr inbounds nuw %[[CAP_TYPE3]], ptr %[[BL_THIS_LOAD]], i32 0, i32 3
// LLVM: %[[BL_BUF1:.+]] = load ptr, ptr %[[BL_BUF1_CAP]], align 8
// LLVM: %[[BL_OFFSET:.+]] = mul nsw i64 %{{.+}}, %[[BL_SIZE1]]
// LLVM: getelementptr {{(inbounds )?}}i64, ptr %[[BL_BUF1]], i64 %[[BL_OFFSET]]
// LLVM: mul nuw i64 8, %[[BL_SIZE1]]
// LLVM: %[[BL_INNER_VLA_SIZE1:.+]] = getelementptr inbounds nuw %[[CAP_TYPE4]], ptr %[[BL_INNER_TEMP]], i32 0, i32 1
// LLVM: store i64 %[[BL_SIZE1]], ptr %[[BL_INNER_VLA_SIZE1]], align 8
// LLVM: %[[BL_INNER_VLA_SIZE2:.+]] = getelementptr inbounds nuw %[[CAP_TYPE4]], ptr %[[BL_INNER_TEMP]], i32 0, i32 3
// LLVM: store i64 %[[BL_SIZE2]], ptr %[[BL_INNER_VLA_SIZE2]], align 8
// LLVM: call void @[[B_LAMBDA_LAMBDA:.+]](ptr {{.*}}%[[BL_INNER_TEMP]])

// LLVM: define {{.*}} void @[[B_LAMBDA_LAMBDA]](
// LLVM: %[[BLL_THIS:.+]] = alloca ptr, align 8
// LLVM: %[[BLL_THIS_LOAD:.+]] = load ptr, ptr %[[BLL_THIS]], align 8
// LLVM: %[[BLL_VLA_SIZE1:.+]] = getelementptr inbounds nuw %[[CAP_TYPE4]], ptr %[[BLL_THIS_LOAD]], i32 0, i32 1
// LLVM: %[[BLL_SIZE1:.+]] = load i64, ptr %[[BLL_VLA_SIZE1]], align 8
// LLVM: %[[BLL_VLA_SIZE2:.+]] = getelementptr inbounds nuw %[[CAP_TYPE4]], ptr %[[BLL_THIS_LOAD]], i32 0, i32 3
// LLVM: %[[BLL_SIZE2:.+]] = load i64, ptr %[[BLL_VLA_SIZE2]], align 8
// LLVM: mul nuw i64 8, %[[BLL_SIZE1]]
// LLVM: %[[BLL_TOTAL:.+]] = mul nuw i64 %[[BLL_SIZE2]], %[[BLL_SIZE1]]
// LLVM: mul nuw i64 8, %[[BLL_TOTAL]]
