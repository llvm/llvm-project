// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=alignment | FileCheck %s

struct S {
  int I;
};

extern S g_S;
extern S array_S[];

// CHECK-LABEL: define i32 @_Z18load_extern_global
int load_extern_global() {
  // FIXME: The IR builder constant-folds the alignment check away to 'true'
  // here, so we never call the diagnostic. This is PR32630.
  // CHECK-NOT: ptrtoint i32* {{.*}} to i32, !nosanitize
  // CHECK: [[I:%.*]] = load i32, i32* getelementptr inbounds (%struct.S, %struct.S* @g_S, i32 0, i32 0), align 4
  // CHECK-NEXT: ret i32 [[I]]
  return g_S.I;
}

// CHECK-LABEL: define i32 @_Z22load_from_extern_array
int load_from_extern_array(int I) {
  // CHECK: [[I:%.*]] = getelementptr inbounds %struct.S, %struct.S* {{.*}}, i32 0, i32 0
  // CHECK-NEXT: [[PTRTOINT:%.*]] = ptrtoint i32* [[I]] to i64, !nosanitize
  // CHECK-NEXT: [[AND:%.*]] = and i64 [[PTRTOINT]], 3, !nosanitize
  // CHECK-NEXT: [[ICMP:%.*]] = icmp eq i64 [[AND]], 0, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]]
  // CHECK: call void @__ubsan_handle_type_mismatch
  return array_S[I].I;
}
