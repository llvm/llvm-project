// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct HasScalarArrayMember {
  int arr[2][2];
  HasScalarArrayMember(const HasScalarArrayMember &);
};

HasScalarArrayMember::HasScalarArrayMember(const HasScalarArrayMember &) = default;

// CIR-LABEL: cir.func {{.*}} @_ZN20HasScalarArrayMemberC2ERKS_(
// CIR-NEXT:    %[[THIS:.*]] = cir.alloca !cir.ptr<!rec_HasScalarArrayMember>
// CIR-NEXT:    %[[OTHER:.*]] = cir.alloca !cir.ptr<!rec_HasScalarArrayMember>
// CIR-NEXT:    cir.store %arg0, %[[THIS]]
// CIR-NEXT:    cir.store %arg1, %[[OTHER]]
// CIR-NEXT:    %[[THIS_LOAD:.*]] = cir.load{{.*}} %[[THIS]]
// CIR-NEXT:    %[[THIS_ARR:.*]] = cir.get_member %[[THIS_LOAD]][0] {name = "arr"}
// CIR-NEXT:    %[[OTHER_LOAD:.*]] = cir.load{{.*}} %[[OTHER]]
// CIR-NEXT:    %[[OTHER_ARR:.*]] = cir.get_member %[[OTHER_LOAD]][0] {name = "arr"}
// CIR-NEXT:    cir.copy %[[OTHER_ARR]] to %[[THIS_ARR]] : !cir.ptr<!cir.array<!cir.array<!s32i x 2> x 2>>
// CIR-NEXT:    cir.return

// LLVM-LABEL: define {{.*}} @_ZN20HasScalarArrayMemberC2ERKS_(
// LLVM-SAME:      ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// LLVM-NEXT:    %[[THIS:.*]] = alloca ptr
// LLVM-NEXT:    %[[OTHER:.*]] = alloca ptr
// LLVM-NEXT:    store ptr %[[ARG0]], ptr %[[THIS]]
// LLVM-NEXT:    store ptr %[[ARG1]], ptr %[[OTHER]]
// LLVM-NEXT:    %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS]]
// LLVM-NEXT:    %[[THIS_ARR:.*]] = getelementptr %struct.HasScalarArrayMember, ptr %[[THIS_LOAD]], i32 0, i32 0
// LLVM-NEXT:    %[[OTHER_LOAD:.*]] = load ptr, ptr %[[OTHER]]
// LLVM-NEXT:    %[[OTHER_ARR:.*]] = getelementptr %struct.HasScalarArrayMember, ptr %[[OTHER_LOAD]], i32 0, i32 0
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr %[[THIS_ARR]], ptr %[[OTHER_ARR]], i32 16, i1 false)
// LLVM-NEXT:    ret void

// OGCG-LABEL: define {{.*}} @_ZN20HasScalarArrayMemberC2ERKS_(
// OGCG-SAME:      ptr {{.*}} %[[ARG0:.*]], ptr {{.*}} %[[ARG1:.*]])
// OGCG-NEXT:  entry:
// OGCG-NEXT:    %[[THIS:.*]] = alloca ptr
// OGCG-NEXT:    %[[OTHER:.*]] = alloca ptr
// OGCG-NEXT:    store ptr %[[ARG0]], ptr %[[THIS]]
// OGCG-NEXT:    store ptr %[[ARG1]], ptr %[[OTHER]]
// OGCG-NEXT:    %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS]]
// OGCG-NEXT:    %[[THIS_ARR:.*]] = getelementptr inbounds nuw %struct.HasScalarArrayMember, ptr %[[THIS_LOAD]], i32 0, i32 0
// OGCG-NEXT:    %[[OTHER_LOAD:.*]] = load ptr, ptr %[[OTHER]]
// OGCG-NEXT:    %[[OTHER_ARR:.*]] = getelementptr inbounds nuw %struct.HasScalarArrayMember, ptr %[[OTHER_LOAD]], i32 0, i32 0
// OGCG-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[THIS_ARR]], ptr {{.*}} %[[OTHER_ARR]], i64 16, i1 false)
// OGCG-NEXT:    ret void
