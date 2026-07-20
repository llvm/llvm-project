// RUN: %clang -S -emit-llvm -O2 --target=x86_64-windows-msvc -fdefined-pointer-subtraction -fms-extensions %s -o - | FileCheck %s

// Check that pointer subtraction isn't nuw/nsv and sdiv isn't exact
// CHECK-LABEL: i64 @sub(ptr noundef %p, ptr noundef %q)
// CHECK-NEXT:  entry:
// CHECK-NEXT:    %sub.ptr.lhs.cast = ptrtoint ptr %p to i64
// CHECK-NEXT:    %sub.ptr.rhs.cast = ptrtoint ptr %q to i64
// CHECK-NEXT:    %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
// CHECK-NEXT:    %sub.ptr.div = sdiv i64 %sub.ptr.sub, 4
// CHECK-NEXT:    ret i64 %sub.ptr.div

__declspec(noinline) long long sub(long* p, long* q) {
  return p - q;
}

