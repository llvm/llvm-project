// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Port of clang/test/CodeGenSYCL/address-space-deduction.cpp. Local variables
// are allocated in the private address space and cast to the generic address
// space, so taking their address yields a generic pointer.
//
// TODO(cir): The static local and the string literals of the original test
// are omitted until CIR supports the SYCL global address space.

[[clang::sycl_external]] void test() {
  int i = 0;
  int *pptr = &i;
  bool is_i_ptr = (pptr == &i);

  int var23 = 23;
  char *cp = (char *)&var23;
  *cp = 41;

  int arr[42];
  char *cpp = (char *)arr;
  *cpp = 43;

  int *aptr = arr + 10;
  if (aptr < arr + sizeof(arr))
    *aptr = 44;
}

// CIR-LABEL: cir.func {{.*}}@_Z4testv()
// CIR:         %[[I:.*]] = cir.alloca "i" {{.*}} : !cir.ptr<!s32i>
// CIR:         %[[PPTR:.*]] = cir.alloca "pptr" {{.*}} : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>>
// CIR:         %[[IS_I_PTR:.*]] = cir.alloca "is_i_ptr" {{.*}} : !cir.ptr<!cir.bool>
// CIR:         %[[VAR23:.*]] = cir.alloca "var23" {{.*}} : !cir.ptr<!s32i>
// CIR:         %[[CP:.*]] = cir.alloca "cp" {{.*}} : !cir.ptr<!cir.ptr<!s8i, target_address_space(4)>>
// CIR:         %[[ARR:.*]] = cir.alloca "arr" {{.*}} : !cir.ptr<!cir.array<!s32i x 42>>
// CIR:         %[[CPP:.*]] = cir.alloca "cpp" {{.*}} : !cir.ptr<!cir.ptr<!s8i, target_address_space(4)>>
// CIR:         %[[APTR:.*]] = cir.alloca "aptr" {{.*}} : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>>
// CIR:         %[[APTR_ASCAST:.*]] = cir.cast address_space %[[APTR]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR:         %[[CPP_ASCAST:.*]] = cir.cast address_space %[[CPP]] : !cir.ptr<!cir.ptr<!s8i, target_address_space(4)>> -> !cir.ptr<!cir.ptr<!s8i, target_address_space(4)>, target_address_space(4)>
// CIR:         %[[ARR_ASCAST:.*]] = cir.cast address_space %[[ARR]] : !cir.ptr<!cir.array<!s32i x 42>> -> !cir.ptr<!cir.array<!s32i x 42>, target_address_space(4)>
// CIR:         %[[CP_ASCAST:.*]] = cir.cast address_space %[[CP]] : !cir.ptr<!cir.ptr<!s8i, target_address_space(4)>> -> !cir.ptr<!cir.ptr<!s8i, target_address_space(4)>, target_address_space(4)>
// CIR:         %[[VAR23_ASCAST:.*]] = cir.cast address_space %[[VAR23]] : !cir.ptr<!s32i> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR:         %[[IS_I_PTR_ASCAST:.*]] = cir.cast address_space %[[IS_I_PTR]] : !cir.ptr<!cir.bool> -> !cir.ptr<!cir.bool, target_address_space(4)>
// CIR:         %[[PPTR_ASCAST:.*]] = cir.cast address_space %[[PPTR]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR:         %[[I_ASCAST:.*]] = cir.cast address_space %[[I]] : !cir.ptr<!s32i> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR:         cir.store {{.*}}, %[[I_ASCAST]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>
// CIR:         cir.store {{.*}} %[[I_ASCAST]], %[[PPTR_ASCAST]] : !cir.ptr<!s32i, target_address_space(4)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR:         %[[PPTR_VAL:.*]] = cir.load {{.*}} %[[PPTR_ASCAST]]
// CIR:         %[[CMP:.*]] = cir.cmp eq %[[PPTR_VAL]], %[[I_ASCAST]] : !cir.ptr<!s32i, target_address_space(4)>
// CIR:         cir.store {{.*}} %[[CMP]], %[[IS_I_PTR_ASCAST]] : !cir.bool, !cir.ptr<!cir.bool, target_address_space(4)>
// CIR:         cir.store {{.*}}, %[[VAR23_ASCAST]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>
// CIR:         %[[VAR23_CHAR:.*]] = cir.cast bitcast %[[VAR23_ASCAST]] : !cir.ptr<!s32i, target_address_space(4)> -> !cir.ptr<!s8i, target_address_space(4)>
// CIR:         cir.store {{.*}} %[[VAR23_CHAR]], %[[CP_ASCAST]]
// CIR:         %[[CP_VAL:.*]] = cir.load deref {{.*}} %[[CP_ASCAST]]
// CIR:         cir.store {{.*}}, %[[CP_VAL]] : !s8i, !cir.ptr<!s8i, target_address_space(4)>
// CIR:         %[[ARR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARR_ASCAST]] : !cir.ptr<!cir.array<!s32i x 42>, target_address_space(4)> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR:         %[[ARR_CHAR:.*]] = cir.cast bitcast %[[ARR_DECAY]] : !cir.ptr<!s32i, target_address_space(4)> -> !cir.ptr<!s8i, target_address_space(4)>
// CIR:         cir.store {{.*}} %[[ARR_CHAR]], %[[CPP_ASCAST]]
// CIR:         %[[CPP_VAL:.*]] = cir.load deref {{.*}} %[[CPP_ASCAST]]
// CIR:         cir.store {{.*}}, %[[CPP_VAL]] : !s8i, !cir.ptr<!s8i, target_address_space(4)>
// CIR:         %[[ARR_DECAY1:.*]] = cir.cast array_to_ptrdecay %[[ARR_ASCAST]]
// CIR:         %[[ADD_PTR:.*]] = cir.ptr_stride %[[ARR_DECAY1]], {{.*}} : (!cir.ptr<!s32i, target_address_space(4)>, !s32i) -> !cir.ptr<!s32i, target_address_space(4)>
// CIR:         cir.store {{.*}} %[[ADD_PTR]], %[[APTR_ASCAST]]
// CIR:         cir.scope {
// CIR:           %[[APTR_VAL:.*]] = cir.load {{.*}} %[[APTR_ASCAST]]
// CIR:           %[[ARR_DECAY2:.*]] = cir.cast array_to_ptrdecay %[[ARR_ASCAST]]
// CIR:           %[[ADD_PTR2:.*]] = cir.ptr_stride %[[ARR_DECAY2]], {{.*}} : (!cir.ptr<!s32i, target_address_space(4)>, !u64i) -> !cir.ptr<!s32i, target_address_space(4)>
// CIR:           %[[CMP2:.*]] = cir.cmp lt %[[APTR_VAL]], %[[ADD_PTR2]] : !cir.ptr<!s32i, target_address_space(4)>
// CIR:           cir.if %[[CMP2]] {
// CIR:             %[[APTR_VAL2:.*]] = cir.load deref {{.*}} %[[APTR_ASCAST]]
// CIR:             cir.store {{.*}}, %[[APTR_VAL2]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>

// LLVM-LABEL: define {{.*}}spir_func void @_Z4testv(
// LLVM: %[[I:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[PPTR:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[IS_I_PTR:.*]] = alloca i8, align 1
// LLVM-NEXT: %[[VAR23:.*]] = alloca i32, align 4
// LLVM-NEXT: %[[CP:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[ARR:.*]] = alloca [42 x i32], align 4
// LLVM-NEXT: %[[CPP:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[APTR:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[APTR_ASCAST:.*]] = addrspacecast ptr %[[APTR]] to ptr addrspace(4)
// LLVM-NEXT: %[[CPP_ASCAST:.*]] = addrspacecast ptr %[[CPP]] to ptr addrspace(4)
// LLVM-NEXT: %[[ARR_ASCAST:.*]] = addrspacecast ptr %[[ARR]] to ptr addrspace(4)
// LLVM-NEXT: %[[CP_ASCAST:.*]] = addrspacecast ptr %[[CP]] to ptr addrspace(4)
// LLVM-NEXT: %[[VAR23_ASCAST:.*]] = addrspacecast ptr %[[VAR23]] to ptr addrspace(4)
// LLVM-NEXT: %[[IS_I_PTR_ASCAST:.*]] = addrspacecast ptr %[[IS_I_PTR]] to ptr addrspace(4)
// LLVM-NEXT: %[[PPTR_ASCAST:.*]] = addrspacecast ptr %[[PPTR]] to ptr addrspace(4)
// LLVM-NEXT: %[[I_ASCAST:.*]] = addrspacecast ptr %[[I]] to ptr addrspace(4)
// LLVM-NEXT: store i32 0, ptr addrspace(4) %[[I_ASCAST]], align 4
// LLVM-NEXT: store ptr addrspace(4) %[[I_ASCAST]], ptr addrspace(4) %[[PPTR_ASCAST]], align 8
// LLVM-NEXT: %[[TMP0:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[PPTR_ASCAST]], align 8
// LLVM-NEXT: %[[TMP1:.*]] = icmp eq ptr addrspace(4) %[[TMP0]], %[[I_ASCAST]]
// LLVM-NEXT: %[[TMP2:.*]] = zext i1 %[[TMP1]] to i8
// LLVM-NEXT: store i8 %[[TMP2]], ptr addrspace(4) %[[IS_I_PTR_ASCAST]], align 1
// LLVM-NEXT: store i32 23, ptr addrspace(4) %[[VAR23_ASCAST]], align 4
// LLVM-NEXT: store ptr addrspace(4) %[[VAR23_ASCAST]], ptr addrspace(4) %[[CP_ASCAST]], align 8
// LLVM-NEXT: %[[TMP3:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[CP_ASCAST]], align 8
// LLVM-NEXT: store i8 41, ptr addrspace(4) %[[TMP3]], align 1
// LLVM-NEXT: %[[TMP4:.*]] = getelementptr i32, ptr addrspace(4) %[[ARR_ASCAST]], i32 0
// LLVM-NEXT: store ptr addrspace(4) %[[TMP4]], ptr addrspace(4) %[[CPP_ASCAST]], align 8
// LLVM-NEXT: %[[TMP5:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[CPP_ASCAST]], align 8
// LLVM-NEXT: store i8 43, ptr addrspace(4) %[[TMP5]], align 1
// LLVM-NEXT: %[[TMP6:.*]] = getelementptr i32, ptr addrspace(4) %[[ARR_ASCAST]], i32 0
// LLVM-NEXT: %[[TMP7:.*]] = getelementptr i32, ptr addrspace(4) %[[TMP6]], i64 10
// LLVM-NEXT: store ptr addrspace(4) %[[TMP7]], ptr addrspace(4) %[[APTR_ASCAST]], align 8
// LLVM-NEXT: br label %[[TMP8:.*]]
// LLVM: [[TMP8]]:
// LLVM-NEXT: %[[TMP9:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[APTR_ASCAST]], align 8
// LLVM-NEXT: %[[TMP10:.*]] = getelementptr i32, ptr addrspace(4) %[[ARR_ASCAST]], i32 0
// LLVM-NEXT: %[[TMP11:.*]] = getelementptr i32, ptr addrspace(4) %[[TMP10]], i64 168
// LLVM-NEXT: %[[TMP12:.*]] = icmp ult ptr addrspace(4) %[[TMP9]], %[[TMP11]]
// LLVM-NEXT: br i1 %[[TMP12]], label %[[TMP13:.*]], label %[[TMP14:.*]]
// LLVM: [[TMP13]]:
// LLVM-NEXT: %[[TMP15:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[APTR_ASCAST]], align 8
// LLVM-NEXT: store i32 44, ptr addrspace(4) %[[TMP15]], align 4
// LLVM-NEXT: br label %[[TMP14]]
// LLVM: [[TMP14]]:
// LLVM-NEXT: br label %[[TMP16:.*]]
// LLVM: [[TMP16]]:
// LLVM-NEXT: ret void

// OGCG-LABEL: define {{.*}}spir_func void @_Z4testv(
// OGCG: %[[I:.*]] = alloca i32, align 4
// OGCG-NEXT: %[[PPTR:.*]] = alloca ptr addrspace(4), align 8
// OGCG-NEXT: %[[IS_I_PTR:.*]] = alloca i8, align 1
// OGCG-NEXT: %[[VAR23:.*]] = alloca i32, align 4
// OGCG-NEXT: %[[CP:.*]] = alloca ptr addrspace(4), align 8
// OGCG-NEXT: %[[ARR:.*]] = alloca [42 x i32], align 4
// OGCG-NEXT: %[[CPP:.*]] = alloca ptr addrspace(4), align 8
// OGCG-NEXT: %[[APTR:.*]] = alloca ptr addrspace(4), align 8
// OGCG-NEXT: %[[I_ASCAST:.*]] = addrspacecast ptr %[[I]] to ptr addrspace(4)
// OGCG-NEXT: %[[PPTR_ASCAST:.*]] = addrspacecast ptr %[[PPTR]] to ptr addrspace(4)
// OGCG-NEXT: %[[IS_I_PTR_ASCAST:.*]] = addrspacecast ptr %[[IS_I_PTR]] to ptr addrspace(4)
// OGCG-NEXT: %[[VAR23_ASCAST:.*]] = addrspacecast ptr %[[VAR23]] to ptr addrspace(4)
// OGCG-NEXT: %[[CP_ASCAST:.*]] = addrspacecast ptr %[[CP]] to ptr addrspace(4)
// OGCG-NEXT: %[[ARR_ASCAST:.*]] = addrspacecast ptr %[[ARR]] to ptr addrspace(4)
// OGCG-NEXT: %[[CPP_ASCAST:.*]] = addrspacecast ptr %[[CPP]] to ptr addrspace(4)
// OGCG-NEXT: %[[APTR_ASCAST:.*]] = addrspacecast ptr %[[APTR]] to ptr addrspace(4)
// OGCG-NEXT: store i32 0, ptr addrspace(4) %[[I_ASCAST]], align 4
// OGCG-NEXT: store ptr addrspace(4) %[[I_ASCAST]], ptr addrspace(4) %[[PPTR_ASCAST]], align 8
// OGCG-NEXT: %[[TMP0:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[PPTR_ASCAST]], align 8
// OGCG-NEXT: %[[CMP:.*]] = icmp eq ptr addrspace(4) %[[TMP0]], %[[I_ASCAST]]
// OGCG-NEXT: %[[STOREDV:.*]] = zext i1 %[[CMP]] to i8
// OGCG-NEXT: store i8 %[[STOREDV]], ptr addrspace(4) %[[IS_I_PTR_ASCAST]], align 1
// OGCG-NEXT: store i32 23, ptr addrspace(4) %[[VAR23_ASCAST]], align 4
// OGCG-NEXT: store ptr addrspace(4) %[[VAR23_ASCAST]], ptr addrspace(4) %[[CP_ASCAST]], align 8
// OGCG-NEXT: %[[TMP1:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[CP_ASCAST]], align 8
// OGCG-NEXT: store i8 41, ptr addrspace(4) %[[TMP1]], align 1
// OGCG-NEXT: %[[ARRAYDECAY:.*]] = getelementptr inbounds [42 x i32], ptr addrspace(4) %[[ARR_ASCAST]], i64 0, i64 0
// OGCG-NEXT: store ptr addrspace(4) %[[ARRAYDECAY]], ptr addrspace(4) %[[CPP_ASCAST]], align 8
// OGCG-NEXT: %[[TMP2:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[CPP_ASCAST]], align 8
// OGCG-NEXT: store i8 43, ptr addrspace(4) %[[TMP2]], align 1
// OGCG-NEXT: %[[ARRAYDECAY1:.*]] = getelementptr inbounds [42 x i32], ptr addrspace(4) %[[ARR_ASCAST]], i64 0, i64 0
// OGCG-NEXT: %[[ADD_PTR:.*]] = getelementptr inbounds i32, ptr addrspace(4) %[[ARRAYDECAY1]], i64 10
// OGCG-NEXT: store ptr addrspace(4) %[[ADD_PTR]], ptr addrspace(4) %[[APTR_ASCAST]], align 8
// OGCG-NEXT: %[[TMP3:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[APTR_ASCAST]], align 8
// OGCG-NEXT: %[[ARRAYDECAY2:.*]] = getelementptr inbounds [42 x i32], ptr addrspace(4) %[[ARR_ASCAST]], i64 0, i64 0
// OGCG-NEXT: %[[ADD_PTR3:.*]] = getelementptr inbounds nuw i32, ptr addrspace(4) %[[ARRAYDECAY2]], i64 168
// OGCG-NEXT: %[[CMP4:.*]] = icmp ult ptr addrspace(4) %[[TMP3]], %[[ADD_PTR3]]
// OGCG-NEXT: br i1 %[[CMP4]], label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// OGCG: [[IF_THEN]]:
// OGCG-NEXT: %[[TMP4:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[APTR_ASCAST]], align 8
// OGCG-NEXT: store i32 44, ptr addrspace(4) %[[TMP4]], align 4
// OGCG-NEXT: br label %[[IF_END]]
// OGCG: [[IF_END]]:
// OGCG-NEXT: ret void
