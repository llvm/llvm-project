// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

template <class T> void consume(T);

void incomplete_array_temporary(long x) {
  consume((int[])(char)x);
}

// CIR-LABEL: cir.func {{.*}} @_Z26incomplete_array_temporaryl
// CIR:         %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!cir.array<!s32i x 1>>
// CIR:         %[[INCOMPLETE:.*]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!cir.array<!s32i x 1>> -> !cir.ptr<!cir.array<!s32i x 0>>
// CIR-NEXT:    %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[INCOMPLETE]] : !cir.ptr<!cir.array<!s32i x 0>> -> !cir.ptr<!s32i>
// CIR-NEXT:    cir.call @{{.*}}(%[[DECAY]])

// LLVM-LABEL: define {{.*}} void @_Z26incomplete_array_temporaryl
// LLVM:         %[[TMP:.*]] = alloca [1 x i32], align 4
// LLVM:         store i32 {{.*}}, ptr {{.*}}, align 4
// LLVM-NEXT:    %[[DECAY:.*]] = getelementptr i32, ptr %[[TMP]], i32 0
// LLVM-NEXT:    call void @{{.*}}(ptr noundef %[[DECAY]])

// OGCG-LABEL: define {{.*}} void @_Z26incomplete_array_temporaryl
// OGCG:         %[[TMP:.*]] = alloca [1 x i32], align 4
// OGCG:         store i32 {{.*}}, ptr %[[TMP]], align 4
// OGCG-NEXT:    %[[DECAY:.*]] = getelementptr inbounds [0 x i32], ptr %[[TMP]], i64 0, i64 0
// OGCG-NEXT:    call void @{{.*}}(ptr noundef %[[DECAY]])
