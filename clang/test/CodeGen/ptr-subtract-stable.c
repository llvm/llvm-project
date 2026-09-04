// RUN: %clang -S -emit-llvm -O2 --target=x86_64-windows-msvc -fdefined-pointer-subtraction -fno-discard-value-names -fms-extensions %s -o - | FileCheck %s

// Check that pointer subtraction isn't nuw/nsv and sdiv isn't exact
// CHECK:       i64 @sub(ptr noundef %[[P:.*]], ptr noundef %[[Q:.*]])
// CHECK-NEXT:  entry:
// CHECK-NEXT:    %[[PI:.*]] = ptrtoint ptr %[[P]] to i64
// CHECK-NEXT:    %[[QI:.*]] = ptrtoint ptr %[[Q]] to i64
// CHECK-NEXT:    %[[SB:.*]] = sub i64 %[[PI]], %[[QI]]
// CHECK-NEXT:    {{.*}}     = sdiv i64 %[[SB]], 4

__declspec(noinline) long long sub(long* p, long* q) {
  return p - q;
}

