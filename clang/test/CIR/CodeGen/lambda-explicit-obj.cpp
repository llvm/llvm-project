// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,CIRONLY --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

struct HasThis {
  int m;
  void func() {
    auto capturesThis = [this](this auto) { (void)m; };
    capturesThis(); }
};

void g(HasThis ht) { ht.func(); }
// LLVM: %[[LAMBDA_TY:.*]] = type { ptr }

// 'g' body'.
// CIR-LABEL: cir.func {{.*}}@_Z1g7HasThis(
// CIR: %[[HT_ALLOCA:.*]] = cir.alloca "ht" align(4) init : !cir.ptr<!rec_HasThis>
// CIR: cir.call @_ZN7HasThis4funcEv(%[[HT_ALLOCA]]) : (!cir.ptr<!rec_HasThis> {{.*}}) -> ()

// LLVM-LABEL: define {{.*}}@_Z1g7HasThis(
// LLVM: %[[HT_ALLOCA:.*]] = alloca %struct.HasThis, align 4
// LLVM: call void @_ZN7HasThis4funcEv(ptr {{.*}}%[[HT_ALLOCA]])

// 'func' body.
// CIR-LABEL: cir.func {{.*}}@_ZN7HasThis4funcEv(
// CIR-SAME: %[[THIS:.*]]: !cir.ptr<!rec_HasThis>
// CIR: %[[COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!rec_anon2E0>
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<!rec_HasThis>>
// CIR: %[[CAPTURE_ALLOCA:.*]] = cir.alloca "capturesThis" align(8) init : !cir.ptr<!rec_anon2E0>
// CIR: %[[TMP_ALLOCA:.*]] = cir.alloca "agg.tmp0" align(8) : !cir.ptr<!rec_anon2E0>
// CIR: cir.store %[[THIS]], %[[THIS_ALLOCA]] : !cir.ptr<!rec_HasThis>, !cir.ptr<!cir.ptr<!rec_HasThis>>
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasThis>>, !cir.ptr<!rec_HasThis>
// CIR: %[[CAPTURE_THIS_GET_MEM:.*]] = cir.get_member %[[CAPTURE_ALLOCA]][0] {name = "this"} : !cir.ptr<!rec_anon2E0> -> !cir.ptr<!cir.ptr<!rec_HasThis>>
// CIR: cir.store align(8) %[[THIS_LOAD]], %[[CAPTURE_THIS_GET_MEM]] : !cir.ptr<!rec_HasThis>, !cir.ptr<!cir.ptr<!rec_HasThis>>
// CIR: cir.copy %[[CAPTURE_ALLOCA]]{{.*}} to %[[TMP_ALLOCA]]{{.*}} : !cir.ptr<!rec_anon2E0>
// CIR: %[[LOAD_TMP:.*]] = cir.load align(8) %[[TMP_ALLOCA]] : !cir.ptr<!rec_anon2E0>, !rec_anon2E0
// CIR: cir.store %[[LOAD_TMP]], %[[COERCE]] : !rec_anon2E0, !cir.ptr<!rec_anon2E0>
// CIR: %[[CAST_COERCE:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!rec_anon2E0> -> !cir.ptr<!cir.ptr<!void>>
// CIR: %[[LOAD_COERCE:.*]] = cir.load %[[CAST_COERCE]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: cir.call @_ZZN7HasThis4funcEvENHUlT_E_clIS1_EEDaS0_(%[[LOAD_COERCE]]) : (!cir.ptr<!void>) -> ()

// LLVM-LABEL: define {{.*}}@_ZN7HasThis4funcEv
// LLVM-SAME: (ptr {{.*}}%[[THIS:.*]])
// OGCG: %[[THIS_ALLOCA:.*]] = alloca ptr, align 8
// CIRONLY: %[[COERCE:.*]] = alloca %[[LAMBDA_TY]], align 8
// CIRONLY: %[[THIS_ALLOCA:.*]] = alloca ptr, align 8
// LLVM: %[[CAPTURE_ALLOCA:.*]] = alloca %[[LAMBDA_TY]], align 8
// LLVM: %[[TMP_ALLOCA:.*]] = alloca %[[LAMBDA_TY]], align 8
// LLVM: store ptr %[[THIS]], ptr %[[THIS_ALLOCA]], align 8
// LLVM: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]], align 8
// LLVM: %[[CAPTURE_THIS_GET_MEM:.*]] = getelementptr inbounds nuw %[[LAMBDA_TY]], ptr %[[CAPTURE_ALLOCA]], i32 0, i32 0
// LLVM: store ptr %[[THIS_LOAD]], ptr %[[CAPTURE_THIS_GET_MEM]], align 8
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[TMP_ALLOCA]], ptr {{.*}}%[[CAPTURE_ALLOCA]], i64 8, i1 false)
// CIRONLY: %[[LOAD_TMP:.*]] = load %[[LAMBDA_TY]], ptr %[[TMP:.*]], align 8
// CIRONLY: store %[[LAMBDA_TY]] %[[LOAD_TMP]], ptr %[[COERCE]], align 8
// OGCG: %[[COERCE:.*]] = getelementptr inbounds nuw %[[LAMBDA_TY]], ptr %[[TMP_ALLOCA]], i32 0, i32 0
// LLVM: %[[LOAD_COERCE:.*]] = load ptr, ptr %[[COERCE]], align 8
// LLVM: call void @_ZZN7HasThis4funcEvENHUlT_E_clIS1_EEDaS0_(ptr %[[LOAD_COERCE]])

// Lambda Body:
// CIR-LABEL: cir.func {{.*}}@_ZZN7HasThis4funcEvENHUlT_E_clIS1_EEDaS0_(
// CIR-SAME: %[[LAMBDA_THIS:.*]]: !cir.ptr<!void>
// CIR: %[[COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!cir.ptr<!void>>
// CIR: cir.store %[[LAMBDA_THIS]], %[[COERCE]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR: %[[CAST_COERCE:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!rec_anon2E0>
// CIR: %[[LOAD_COERCE:.*]] = cir.load %[[CAST_COERCE]] : !cir.ptr<!rec_anon2E0>, !rec_anon2E0
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca "" align(8) init : !cir.ptr<!rec_anon2E0>
// CIR: cir.store %[[LOAD_COERCE]], %[[THIS_ALLOCA]] : !rec_anon2E0, !cir.ptr<!rec_anon2E0>
// CIR: %[[GET_HAS_THIS:.*]] = cir.get_member %[[THIS_ALLOCA]][0] {name = "this"} : !cir.ptr<!rec_anon2E0> -> !cir.ptr<!cir.ptr<!rec_HasThis>>
// CIR: %[[LOAD_HAS_THIS:.*]] = cir.load align(8) %[[GET_HAS_THIS]] : !cir.ptr<!cir.ptr<!rec_HasThis>>, !cir.ptr<!rec_HasThis>
// CIR: cir.get_member %[[LOAD_HAS_THIS]][0] {name = "m"} : !cir.ptr<!rec_HasThis> -> !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}@_ZZN7HasThis4funcEvENHUlT_E_clIS1_EEDaS0_(
// LLVM-SAME: ptr %[[LAMBDA_THIS:.*]])
// CIRONLY: %[[COERCE:.*]] = alloca ptr, align 8
// OGCG: %[[LAMBDA_ALLOCA:.*]] = alloca %[[LAMBDA_TY]]
// OGCG: %[[COERCE:.*]] = getelementptr inbounds nuw %[[LAMBDA_TY]], ptr %[[LAMBDA_ALLOCA]], i32 0, i32 0

// LLVM: store ptr %[[LAMBDA_THIS]], ptr %[[COERCE]], align 8

// CIRONLY: %[[LOAD_COERCE:.*]] = load %[[LAMBDA_TY]], ptr %[[COERCE]], align 8
// CIRONLY: %[[THIS_ALLOCA:.*]] = alloca %[[LAMBDA_TY]], align 8
// CIRONLY: store %[[LAMBDA_TY]] %[[LOAD_COERCE]], ptr %[[THIS_ALLOCA]], align 8
// CIRONLY: %[[GET_HAS_THIS:.*]] = getelementptr inbounds nuw %[[LAMBDA_TY]], ptr %[[THIS_ALLOCA]], i32 0, i32 0

// OGCG: %[[GET_HAS_THIS:.*]] = getelementptr inbounds nuw %[[LAMBDA_TY]], ptr %[[LAMBDA_ALLOCA]], i32 0, i32 0

// LLVM: %[[LOAD_HAS_THIS:.*]] = load ptr, ptr %[[GET_HAS_THIS]], align 8
// LLVM: getelementptr inbounds nuw %struct.HasThis, ptr %[[LOAD_HAS_THIS]], i32 0, i32 0
