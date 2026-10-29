// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -std=c++17 -Wno-c99-extensions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -std=c++17 -Wno-c99-extensions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -std=c++17 -Wno-c99-extensions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Temporaries in SYCL device code are allocated in the private address space
// and cast to the generic address space. There is no classic CodeGenSYCL
// counterpart for these cases.

void use(int *p);
void use_const(const int *p);

struct S {
  int a;
  int b;
};
void take_ref(const S &s);

// Temporary materialized for a reference binding.
[[clang::sycl_external]] void ref_temp() {
  const int &r = 42;
  use_const(&r);
}

// Aggregate temporary passed by reference.
[[clang::sycl_external]] void agg_temp() { take_ref(S{1, 2}); }

// Array temporary that decays to a pointer.
[[clang::sycl_external]] void array_temp() { use((int[]){3, 4}); }

// Local captured by reference in a lambda.
[[clang::sycl_external]] void lambda_ref() {
  int x = 0;
  auto l = [&] { use(&x); };
  l();
}

// NRVO variable lives in the return slot, cast to the generic address space.
[[clang::sycl_external]] S nrvo() {
  S s{};
  use(&s.a);
  return s;
}

// CIR-LABEL: cir.func {{.*}}@_Z8ref_tempv(
// CIR-NEXT: %[[REF_TMP0:.*]] = cir.alloca "ref.tmp0" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[REF_TMP0_ASCAST:.*]] = cir.cast address_space %[[REF_TMP0]] : !cir.ptr<!s32i> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[R:.*]] = cir.alloca "r" align(8) init const : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>>
// CIR-NEXT: %[[R_ASCAST:.*]] = cir.cast address_space %[[R]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR-NEXT: %[[TMP0:.*]] = cir.const #cir.int<42> : !s32i
// CIR-NEXT: cir.store align(4) %[[TMP0]], %[[REF_TMP0_ASCAST]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: cir.store align(8) %[[REF_TMP0_ASCAST]], %[[R_ASCAST]] : !cir.ptr<!s32i, target_address_space(4)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR-NEXT: %[[TMP1:.*]] = cir.load %[[R_ASCAST]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>, !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: cir.call @_Z9use_constPKi(%[[TMP1]]) cc(spir_function) nothrow nounwind {convergent} : (!cir.ptr<!s32i, target_address_space(4)> {llvm.noundef}) -> ()
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z8agg_tempv(
// CIR-NEXT: %[[REF_TMP0:.*]] = cir.alloca "ref.tmp0" align(4) : !cir.ptr<!rec_S>
// CIR-NEXT: %[[REF_TMP0_ASCAST:.*]] = cir.cast address_space %[[REF_TMP0]] : !cir.ptr<!rec_S> -> !cir.ptr<!rec_S, target_address_space(4)>
// CIR-NEXT: %[[TMP0:.*]] = cir.get_member %[[REF_TMP0_ASCAST]][0] {name = "a"} : !cir.ptr<!rec_S, target_address_space(4)> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[TMP1:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT: cir.store align(4) %[[TMP1]], %[[TMP0]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[TMP2:.*]] = cir.get_member %[[REF_TMP0_ASCAST]][1] {name = "b"} : !cir.ptr<!rec_S, target_address_space(4)> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[TMP3:.*]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT: cir.store align(4) %[[TMP3]], %[[TMP2]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: cir.call @_Z8take_refRK1S(%[[REF_TMP0_ASCAST]]) cc(spir_function) nothrow nounwind {convergent} : (!cir.ptr<!rec_S, target_address_space(4)> {llvm.align = 4 : i64, llvm.dereferenceable = 8 : i64, llvm.noundef}) -> ()
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z10array_tempv(
// CIR-NEXT: %[[REF_TMP0:.*]] = cir.alloca "ref.tmp0" align(4) : !cir.ptr<!cir.array<!s32i x 2>>
// CIR-NEXT: %[[REF_TMP0_ASCAST:.*]] = cir.cast address_space %[[REF_TMP0]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!cir.array<!s32i x 2>, target_address_space(4)>
// CIR-NEXT: %[[TMP0:.*]] = cir.cast array_to_ptrdecay %[[REF_TMP0_ASCAST]] : !cir.ptr<!cir.array<!s32i x 2>, target_address_space(4)> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[TMP1:.*]] = cir.const #cir.int<3> : !s32i
// CIR-NEXT: cir.store align(4) %[[TMP1]], %[[TMP0]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[TMP2:.*]] = cir.const #cir.int<1> : !s64i
// CIR-NEXT: %[[TMP3:.*]] = cir.ptr_stride %[[TMP0]], %[[TMP2]] : (!cir.ptr<!s32i, target_address_space(4)>, !s64i) -> !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[TMP4:.*]] = cir.const #cir.int<4> : !s32i
// CIR-NEXT: cir.store align(4) %[[TMP4]], %[[TMP3]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[TMP5:.*]] = cir.cast array_to_ptrdecay %[[REF_TMP0_ASCAST]] : !cir.ptr<!cir.array<!s32i x 2>, target_address_space(4)> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: cir.call @_Z3usePi(%[[TMP5]]) cc(spir_function) nothrow nounwind {convergent} : (!cir.ptr<!s32i, target_address_space(4)> {llvm.noundef}) -> ()
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z10lambda_refv(
// CIR-NEXT: %[[X:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT: %[[L:.*]] = cir.alloca "l" align(8) init : !cir.ptr<!rec_anon2E0>
// CIR-NEXT: %[[L_ASCAST:.*]] = cir.cast address_space %[[L]] : !cir.ptr<!rec_anon2E0> -> !cir.ptr<!rec_anon2E0, target_address_space(4)>
// CIR-NEXT: %[[X_ASCAST:.*]] = cir.cast address_space %[[X]] : !cir.ptr<!s32i> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[TMP0:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: cir.store align(4) %[[TMP0]], %[[X_ASCAST]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: %[[TMP1:.*]] = cir.get_member %[[L_ASCAST]][0] {name = "x"} : !cir.ptr<!rec_anon2E0, target_address_space(4)> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR-NEXT: cir.store align(8) %[[X_ASCAST]], %[[TMP1]] : !cir.ptr<!s32i, target_address_space(4)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR-NEXT: cir.call @_ZZ10lambda_refvENKUlvE_clEv(%[[L_ASCAST]]) cc(spir_function) nothrow nounwind {convergent} : (!cir.ptr<!rec_anon2E0, target_address_space(4)> {llvm.align = 8 : i64, llvm.dereferenceable_or_null = 8 : i64, llvm.noundef}) -> ()
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z4nrvov(
// CIR-NEXT: %[[__RETVAL:.*]] = cir.alloca "__retval" align(4) init : !cir.ptr<!rec_S>
// CIR-NEXT: %[[__RETVAL_ASCAST:.*]] = cir.cast address_space %[[__RETVAL]] : !cir.ptr<!rec_S> -> !cir.ptr<!rec_S, target_address_space(4)>
// CIR-NEXT: %[[TMP0:.*]] = cir.const #cir.zero : !rec_S
// CIR-NEXT: cir.store align(4) %[[TMP0]], %[[__RETVAL_ASCAST]] : !rec_S, !cir.ptr<!rec_S, target_address_space(4)>
// CIR-NEXT: %[[TMP1:.*]] = cir.get_member %[[__RETVAL_ASCAST]][0] {name = "a"} : !cir.ptr<!rec_S, target_address_space(4)> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT: cir.call @_Z3usePi(%[[TMP1]]) cc(spir_function) nothrow nounwind {convergent} : (!cir.ptr<!s32i, target_address_space(4)> {llvm.noundef}) -> ()
// CIR-NEXT: %[[TMP2:.*]] = cir.load %[[__RETVAL]] : !cir.ptr<!rec_S>, !rec_S
// CIR-NEXT: cir.return %[[TMP2]] : !rec_S

// LLVM-LABEL: define {{.*}}spir_func void @_Z8ref_tempv(
// LLVM: %[[R:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[R_ASCAST:.*]] = addrspacecast ptr %[[R]] to ptr addrspace(4)
// LLVM-NEXT: %[[REF_TMP:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[REF_TMP_ASCAST:.*]] = addrspacecast ptr %[[REF_TMP]] to ptr addrspace(4)
// LLVM-NEXT: store i32 42, ptr addrspace(4) %[[R_ASCAST]], align 4
// LLVM-NEXT: store ptr addrspace(4) %[[R_ASCAST]], ptr addrspace(4) %[[REF_TMP_ASCAST]], align 8
// LLVM-NEXT: %[[TMP0:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[REF_TMP_ASCAST]], align 8
// LLVM-NEXT: call spir_func void @_Z9use_constPKi(ptr addrspace(4) noundef %[[TMP0]])
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z8agg_tempv(
// LLVM: %[[REF_TMP:.*]] = alloca %[[TMP0:.*]], align 4
// LLVM-NEXT: %[[REF_TMP_ASCAST:.*]] = addrspacecast ptr %[[REF_TMP]] to ptr addrspace(4)
// LLVM-NEXT: %[[TMP1:.*]] = getelementptr inbounds nuw %[[TMP0]], ptr addrspace(4) %[[REF_TMP_ASCAST]], i32 0, i32 0
// LLVM-NEXT: store i32 1, ptr addrspace(4) %[[TMP1]], align 4
// LLVM-NEXT: %[[TMP2:.*]] = getelementptr inbounds nuw %[[TMP0]], ptr addrspace(4) %[[REF_TMP_ASCAST]], i32 0, i32 1
// LLVM-NEXT: store i32 2, ptr addrspace(4) %[[TMP2]], align 4
// LLVM-NEXT: call spir_func void @_Z8take_refRK1S(ptr addrspace(4) noundef align 4 dereferenceable(8) %[[REF_TMP_ASCAST]])
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z10array_tempv(
// LLVM: %[[REF_TMP:.*]] = alloca [2 x i32], align 4
// LLVM-NEXT: %[[REF_TMP_ASCAST:.*]] = addrspacecast ptr %[[REF_TMP]] to ptr addrspace(4)
// LLVM-NEXT: %[[TMP0:.*]] = getelementptr i32, ptr addrspace(4) %[[REF_TMP_ASCAST]], i32 0
// LLVM-NEXT: store i32 3, ptr addrspace(4) %[[TMP0]], align 4
// LLVM-NEXT: %[[TMP1:.*]] = getelementptr i32, ptr addrspace(4) %[[TMP0]], i64 1
// LLVM-NEXT: store i32 4, ptr addrspace(4) %[[TMP1]], align 4
// LLVM-NEXT: %[[TMP2:.*]] = getelementptr i32, ptr addrspace(4) %[[REF_TMP_ASCAST]], i32 0
// LLVM-NEXT: call spir_func void @_Z3usePi(ptr addrspace(4) noundef %[[TMP2]])
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z10lambda_refv(
// LLVM: %[[X:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[L:.*]] = alloca %[[TMP0:.*]], align 8
// LLVM-NEXT: %[[L_ASCAST:.*]] = addrspacecast ptr %[[L]] to ptr addrspace(4)
// LLVM-NEXT: %[[X_ASCAST:.*]] = addrspacecast ptr %[[X]] to ptr addrspace(4)
// LLVM-NEXT: store i32 0, ptr addrspace(4) %[[X_ASCAST]], align 4
// LLVM-NEXT: %[[TMP1:.*]] = getelementptr inbounds nuw %[[TMP0]], ptr addrspace(4) %[[L_ASCAST]], i32 0, i32 0
// LLVM-NEXT: store ptr addrspace(4) %[[X_ASCAST]], ptr addrspace(4) %[[TMP1]], align 8
// LLVM-NEXT: call spir_func void @_ZZ10lambda_refvENKUlvE_clEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %[[L_ASCAST]])
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_ZZ10lambda_refvENKUlvE_clEv(
// LLVM-SAME: ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %[[THIS:.*]])
// LLVM: %[[THIS_ADDR:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[THIS_ADDR_ASCAST:.*]] = addrspacecast ptr %[[THIS_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(4) %[[THIS]], ptr addrspace(4) %[[THIS_ADDR_ASCAST]], align 8
// LLVM-NEXT: %[[TMP0:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[THIS_ADDR_ASCAST]], align 8
// LLVM-NEXT: %[[TMP1:.*]] = getelementptr inbounds nuw %[[TMP2:.*]], ptr addrspace(4) %[[TMP0]], i32 0, i32 0
// LLVM-NEXT: %[[TMP3:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[TMP1]], align 8
// LLVM-NEXT: call spir_func void @_Z3usePi(ptr addrspace(4) noundef %[[TMP3]])
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func %struct.S @_Z4nrvov(
// LLVM: %[[TMP0:.*]] = alloca %[[TMP1:.*]], align 4
// LLVM-NEXT: %[[TMP2:.*]] = addrspacecast ptr %[[TMP0]] to ptr addrspace(4)
// LLVM-NEXT: store %[[TMP1]] zeroinitializer, ptr addrspace(4) %[[TMP2]], align 4
// LLVM-NEXT: %[[TMP3:.*]] = getelementptr inbounds nuw %[[TMP1]], ptr addrspace(4) %[[TMP2]], i32 0, i32 0
// LLVM-NEXT: call spir_func void @_Z3usePi(ptr addrspace(4) noundef %[[TMP3]])
// LLVM-NEXT: %[[TMP4:.*]] = load %[[TMP1]], ptr %[[TMP0]], align 4
// LLVM-NEXT: ret %[[TMP1]] %[[TMP4]]

// OGCG-LABEL: define {{.*}}spir_func void @_Z8ref_tempv(
// OGCG: %[[R:.*]] = alloca ptr addrspace(4), align 8
// OGCG-NEXT: %[[REF_TMP:.*]] = alloca i32, align 4
// OGCG-NEXT: %[[R_ASCAST:.*]] = addrspacecast ptr %[[R]] to ptr addrspace(4)
// OGCG-NEXT: %[[REF_TMP_ASCAST:.*]] = addrspacecast ptr %[[REF_TMP]] to ptr addrspace(4)
// OGCG-NEXT: store i32 42, ptr addrspace(4) %[[REF_TMP_ASCAST]], align 4
// OGCG-NEXT: store ptr addrspace(4) %[[REF_TMP_ASCAST]], ptr addrspace(4) %[[R_ASCAST]], align 8
// OGCG-NEXT: %[[TMP0:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[R_ASCAST]], align 8, !align !1
// OGCG-NEXT: call spir_func void @_Z9use_constPKi(ptr addrspace(4) noundef %[[TMP0]])
// OGCG-NEXT: ret void

// OGCG-LABEL: define {{.*}}spir_func void @_Z8agg_tempv(
// OGCG: %[[REF_TMP:.*]] = alloca %[[STRUCT_S:.*]], align 4
// OGCG-NEXT: %[[REF_TMP_ASCAST:.*]] = addrspacecast ptr %[[REF_TMP]] to ptr addrspace(4)
// OGCG-NEXT: %[[A:.*]] = getelementptr inbounds nuw %[[STRUCT_S]], ptr addrspace(4) %[[REF_TMP_ASCAST]], i32 0, i32 0
// OGCG-NEXT: store i32 1, ptr addrspace(4) %[[A]], align 4
// OGCG-NEXT: %[[B:.*]] = getelementptr inbounds nuw %[[STRUCT_S]], ptr addrspace(4) %[[REF_TMP_ASCAST]], i32 0, i32 1
// OGCG-NEXT: store i32 2, ptr addrspace(4) %[[B]], align 4
// OGCG-NEXT: call spir_func void @_Z8take_refRK1S(ptr addrspace(4) noundef align 4 dereferenceable(8) %[[REF_TMP_ASCAST]])
// OGCG-NEXT: ret void

// OGCG-LABEL: define {{.*}}spir_func void @_Z10array_tempv(
// OGCG: %[[REF_TMP:.*]] = alloca [2 x i32], align 4
// OGCG-NEXT: %[[REF_TMP_ASCAST:.*]] = addrspacecast ptr %[[REF_TMP]] to ptr addrspace(4)
// OGCG-NEXT: store i32 3, ptr addrspace(4) %[[REF_TMP_ASCAST]], align 4
// OGCG-NEXT: %[[ARRAYINIT_ELEMENT:.*]] = getelementptr inbounds i32, ptr addrspace(4) %[[REF_TMP_ASCAST]], i64 1
// OGCG-NEXT: store i32 4, ptr addrspace(4) %[[ARRAYINIT_ELEMENT]], align 4
// OGCG-NEXT: %[[ARRAYDECAY:.*]] = getelementptr inbounds [2 x i32], ptr addrspace(4) %[[REF_TMP_ASCAST]], i64 0, i64 0
// OGCG-NEXT: call spir_func void @_Z3usePi(ptr addrspace(4) noundef %[[ARRAYDECAY]])
// OGCG-NEXT: ret void

// OGCG-LABEL: define {{.*}}spir_func void @_Z10lambda_refv(
// OGCG: %[[X:.*]] = alloca i32, align 4
// OGCG-NEXT: %[[L:.*]] = alloca %[[CLASS_ANON:.*]], align 8
// OGCG-NEXT: %[[X_ASCAST:.*]] = addrspacecast ptr %[[X]] to ptr addrspace(4)
// OGCG-NEXT: %[[L_ASCAST:.*]] = addrspacecast ptr %[[L]] to ptr addrspace(4)
// OGCG-NEXT: store i32 0, ptr addrspace(4) %[[X_ASCAST]], align 4
// OGCG-NEXT: %[[TMP0:.*]] = getelementptr inbounds nuw %[[CLASS_ANON]], ptr addrspace(4) %[[L_ASCAST]], i32 0, i32 0
// OGCG-NEXT: store ptr addrspace(4) %[[X_ASCAST]], ptr addrspace(4) %[[TMP0]], align 8
// OGCG-NEXT: call spir_func void @_ZZ10lambda_refvENKUlvE_clEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %[[L_ASCAST]])
// OGCG-NEXT: ret void

// OGCG-LABEL: define {{.*}}spir_func void @_ZZ10lambda_refvENKUlvE_clEv(
// OGCG-SAME: ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %[[THIS:.*]])
// OGCG: %[[THIS_ADDR:.*]] = alloca ptr addrspace(4), align 8
// OGCG-NEXT: %[[THIS_ADDR_ASCAST:.*]] = addrspacecast ptr %[[THIS_ADDR]] to ptr addrspace(4)
// OGCG-NEXT: store ptr addrspace(4) %[[THIS]], ptr addrspace(4) %[[THIS_ADDR_ASCAST]], align 8
// OGCG-NEXT: %[[THIS1:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[THIS_ADDR_ASCAST]], align 8
// OGCG-NEXT: %[[TMP0:.*]] = getelementptr inbounds nuw %[[CLASS_ANON:.*]], ptr addrspace(4) %[[THIS1]], i32 0, i32 0
// OGCG-NEXT: %[[TMP1:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[TMP0]], align 8, !align !1
// OGCG-NEXT: call spir_func void @_Z3usePi(ptr addrspace(4) noundef %[[TMP1]])
// OGCG-NEXT: ret void

// OGCG-LABEL: define {{.*}}spir_func void @_Z4nrvov(
// OGCG-SAME: ptr dead_on_unwind noalias writable sret(%[[STRUCT_S:.*]]) align 4 %[[AGG_RESULT:.*]])
// OGCG: %[[AGG_RESULT_ASCAST:.*]] = addrspacecast ptr %[[AGG_RESULT]] to ptr addrspace(4)
// OGCG-NEXT: call void @llvm.memset.p4.i64(ptr addrspace(4) align 4 %[[AGG_RESULT_ASCAST]], i8 0, i64 8, i1 false)
// OGCG-NEXT: %[[A:.*]] = getelementptr inbounds nuw %[[STRUCT_S]], ptr addrspace(4) %[[AGG_RESULT_ASCAST]], i32 0, i32 0
// OGCG-NEXT: call spir_func void @_Z3usePi(ptr addrspace(4) noundef %[[A]])
// OGCG-NEXT: ret void
