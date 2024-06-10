// RUN: %clang_cc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %s | FileCheck %s

int main() { 
  int ret = 0; 
  ret += __builtin_cpu_supports("vsx");     // Test reading `vsx` information from the system variable `_system_configuration`.
  ret += __builtin_cpu_supports("htm");     // Test getting `htm` information from the function call `getsystemcfg`
  ret += __builtin_cpu_supports("cellbe");  // The test always returns false for the feature 'cellbe.
  ret += __builtin_cpu_supports("power4");  // The test always returns false for the feature `power4`.
  ret += __builtin_cpu_supports("fpu");     // The test always returns true for the feature `fpu`.
  ret += __builtin_cpu_supports("mma");     // Test getting `mma` information from the function call `getsystemcfg`.
  return ret;
}

// CHECK:     @_system_configuration = external global { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i64, i32, i8, i8, i8, i8, i32, i32, i16, i16, [3 x i32], i32 }
// CHECK-EMPTY: 
// CHECK-NEXT: ; Function Attrs: noinline nounwind optnone
// CHECK-NEXT: define i32 @main() #0 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %retval = alloca i32, align 4
// CHECK-NEXT:   %ret = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %retval, align 4
// CHECK-NEXT:   store i32 0, ptr %ret, align 4
// CHECK-NEXT:   %0 = load i32, ptr getelementptr inbounds ({ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i64, i32, i8, i8, i8, i8, i32, i32, i16, i16, [3 x i32], i32 }, ptr @_system_configuration, i32 0, i32 46), align 4
// CHECK-NEXT:   %1 = icmp ugt i32 %0, 1
// CHECK-NEXT:   %conv = zext i1 %1 to i32
// CHECK-NEXT:   %2 = load i32, ptr %ret, align 4
// CHECK-NEXT:   %add = add nsw i32 %2, %conv
// CHECK-NEXT:   store i32 %add, ptr %ret, align 4
// CHECK-NEXT:   %3 = call i64 @getsystemcfg(i32 59)
// CHECK-NEXT:   %4 = icmp ugt i64 %3, 0
// CHECK-NEXT:   %conv1 = zext i1 %4 to i32
// CHECK-NEXT:   %5 = load i32, ptr %ret, align 4
// CHECK-NEXT:   %add2 = add nsw i32 %5, %conv1
// CHECK-NEXT:   store i32 %add2, ptr %ret, align 4
// CHECK-NEXT:   %6 = load i32, ptr %ret, align 4
// CHECK-NEXT:   %add3 = add nsw i32 %6, 0
// CHECK-NEXT:   store i32 %add3, ptr %ret, align 4
// CHECK-NEXT:   %7 = load i32, ptr %ret, align 4
// CHECK-NEXT:   %add4 = add nsw i32 %7, 1
// CHECK-NEXT:   store i32 %add4, ptr %ret, align 4
// CHECK-NEXT:   %8 = load i32, ptr %ret, align 4
// CHECK-NEXT:   %add5 = add nsw i32 %8, 1
// CHECK-NEXT:   store i32 %add5, ptr %ret, align 4
// CHECK-NEXT:   %9 = call i64 @getsystemcfg(i32 62)
// CHECK-NEXT:   %10 = icmp ugt i64 %9, 0
// CHECK-NEXT:   %conv6 = zext i1 %10 to i32
// CHECK-NEXT:   %11 = load i32, ptr %ret, align 4
// CHECK-NEXT:   %add7 = add nsw i32 %11, %conv6
// CHECK-NEXT:   store i32 %add7, ptr %ret, align 4
// CHECK-NEXT:   %12 = load i32, ptr %ret, align 4
// CHECK-NEXT:   ret i32 %12
// CHECK-NEXT: }
// CHECK-EMPTY: 
// CHECK-NEXT: declare i64 @getsystemcfg(i32)
