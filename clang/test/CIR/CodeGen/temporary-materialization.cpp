// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

int make_int();

int test() {
  const int &x = make_int();
  return x;
}

//      CIR: cir.func {{.*}} @_Z4testv()
//      CIR:   %[[TEMP_SLOT:.*]] = cir.alloca "ref.tmp0" {{.*}} init : !cir.ptr<!s32i>
// CIR-NEXT:   %[[X:.*]] = cir.alloca "x" {{.*}} init const : !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:   %[[TEMP_VALUE:.*]] = cir.call @_Z8make_intv() : () -> (!s32i {llvm.noundef})
// CIR-NEXT:   cir.store{{.*}} %[[TEMP_VALUE]], %[[TEMP_SLOT]]
// CIR-NEXT:   cir.store{{.*}} %[[TEMP_SLOT]], %[[X]]

// LLVM: define {{.*}} i32 @_Z4testv()
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   %[[TEMP_SLOT:.*]] = alloca i32
// LLVM:   %[[X:.*]] = alloca ptr
// LLVM:   %[[TEMP_VALUE:.*]] = call noundef i32 @_Z8make_intv()
// LLVM:   store i32 %[[TEMP_VALUE]], ptr %[[TEMP_SLOT]]
// LLVM:   store ptr %[[TEMP_SLOT]], ptr %[[X]]

// OGCG: define {{.*}} i32 @_Z4testv()
// OGCG:   %[[X:.*]] = alloca ptr
// OGCG:   %[[TEMP_SLOT:.*]] = alloca i32
// OGCG:   %[[TEMP_VALUE:.*]] = call noundef i32 @_Z8make_intv()
// OGCG:   store i32 %[[TEMP_VALUE]], ptr %[[TEMP_SLOT]]
// OGCG:   store ptr %[[TEMP_SLOT]], ptr %[[X]]

int test_scoped() {
  int x = make_int();
  {
    const int &y = make_int();
    x = y;
  }
  return x;
}

//      CIR: cir.func {{.*}} @_Z11test_scopedv()
//      CIR:   %[[X:.*]] = cir.alloca "x" {{.*}} init : !cir.ptr<!s32i>
//      CIR:   cir.scope {
// CIR-NEXT:     %[[TEMP_SLOT:.*]] = cir.alloca "ref.tmp0" {{.*}} init : !cir.ptr<!s32i>
// CIR-NEXT:     %[[Y_ADDR:.*]] = cir.alloca "y" {{.*}} init const : !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:     %[[TEMP_VALUE:.*]] = cir.call @_Z8make_intv() : () -> (!s32i {llvm.noundef})
// CIR-NEXT:     cir.store{{.*}} %[[TEMP_VALUE]], %[[TEMP_SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:     cir.store{{.*}} %[[TEMP_SLOT]], %[[Y_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:     %[[Y_REF:.*]] = cir.load %[[Y_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:     %[[Y_VALUE:.*]] = cir.load{{.*}} %[[Y_REF]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:     cir.store{{.*}} %[[Y_VALUE]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   }

// LLVM: define {{.*}} i32 @_Z11test_scopedv()
// LLVM:   %[[TEMP_SLOT:.*]] = alloca i32
// LLVM:   %[[Y_ADDR:.*]] = alloca ptr
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   %[[X:.*]] = alloca i32
// LLVM:   %[[TEMP_VALUE1:.*]] = call noundef i32 @_Z8make_intv()
// LLVM:   store i32 %[[TEMP_VALUE1]], ptr %[[X]]
// LLVM:   br label %[[SCOPE_LABEL:.*]]
// LLVM: [[SCOPE_LABEL]]:
// LLVM:   %[[TEMP_VALUE2:.*]] = call noundef i32 @_Z8make_intv()
// LLVM:   store i32 %[[TEMP_VALUE2]], ptr %[[TEMP_SLOT]]
// LLVM:   store ptr %[[TEMP_SLOT]], ptr %[[Y_ADDR]]
// LLVM:   %[[Y_REF:.*]] = load ptr, ptr %[[Y_ADDR]]
// LLVM:   %[[Y_VALUE:.*]] = load i32, ptr %[[Y_REF]]
// LLVM:   store i32 %[[Y_VALUE]], ptr %[[X]]

// OGCG: define {{.*}} i32 @_Z11test_scopedv()
// OGCG:   %[[X:.*]] = alloca i32
// OGCG:   %[[Y_ADDR:.*]] = alloca ptr
// OGCG:   %[[TEMP_SLOT:.*]] = alloca i32
// OGCG:   %[[TEMP_VALUE1:.*]] = call noundef i32 @_Z8make_intv()
// OGCG:   store i32 %[[TEMP_VALUE1]], ptr %[[X]]
// OGCG:   %[[TEMP_VALUE2:.*]] = call noundef i32 @_Z8make_intv()
// OGCG:   store i32 %[[TEMP_VALUE2]], ptr %[[TEMP_SLOT]]
// OGCG:   store ptr %[[TEMP_SLOT]], ptr %[[Y_ADDR]]
// OGCG:   %[[Y_REF:.*]] = load ptr, ptr %[[Y_ADDR]]
// OGCG:   %[[Y_VALUE:.*]] = load i32, ptr %[[Y_REF]]
// OGCG:   store i32 %[[Y_VALUE]], ptr %[[X]]

int test_rvalue_reference(int&& ri) {
  return static_cast<int&&>(ri + 1);
}

//      CIR: cir.func {{.*}} @_Z21test_rvalue_referenceOi({{.*}})
//      CIR:   %[[RI:.*]] = cir.alloca "ri" {{.*}} init const : !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR-NEXT:   %[[TEMP_SLOT:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!s32i>
// CIR-NEXT:   cir.store{{.*}} {{.*}}, %[[RI]]
// CIR-NEXT:   %[[RI_REF:.*]] = cir.load %[[RI]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RI_VALUE:.*]] = cir.load{{.*}} %[[RI_REF]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:   %[[ADD:.*]] = cir.add nsw %[[RI_VALUE]], %[[ONE]] : !s32i
// CIR-NEXT:   cir.store{{.*}} %[[ADD]], %[[TEMP_SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[TEMP_VALUE:.*]] = cir.load{{.*}} %[[TEMP_SLOT]] : !cir.ptr<!s32i>, !s32i

// LLVM: define {{.*}} i32 @_Z21test_rvalue_referenceOi({{.*}})
// LLVM:   %[[RI:.*]] = alloca ptr
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   %[[TEMP_SLOT:.*]] = alloca i32
// LLVM:   store ptr {{.*}}, ptr %[[RI]]
// LLVM:   %[[RI_REF:.*]] = load ptr, ptr %[[RI]]
// LLVM:   %[[RI_VALUE:.*]] = load i32, ptr %[[RI_REF]]
// LLVM:   %[[ADD:.*]] = add nsw i32 %[[RI_VALUE]], 1
// LLVM:   store i32 %[[ADD]], ptr %[[TEMP_SLOT]]
// LLVM:   %[[TEMP_VALUE:.*]] = load i32, ptr %[[TEMP_SLOT]]

// OGCG: define {{.*}} i32 @_Z21test_rvalue_referenceOi({{.*}})
// OGCG:   %[[RI:.*]] = alloca ptr
// OGCG:   %[[TEMP_SLOT:.*]] = alloca i32
// OGCG:   store ptr {{.*}}, ptr %[[RI]]
// OGCG:   %[[RI_REF:.*]] = load ptr, ptr %[[RI]]
// OGCG:   %[[RI_VALUE:.*]] = load i32, ptr %[[RI_REF]]
// OGCG:   %[[ADD:.*]] = add nsw i32 %[[RI_VALUE]], 1
// OGCG:   store i32 %[[ADD]], ptr %[[TEMP_SLOT]]
// OGCG:   %[[TEMP_VALUE:.*]] = load i32, ptr %[[TEMP_SLOT]]

