// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-I386
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-X64

#if defined(__i386__)
long test__readfsdword(unsigned long Offset) {
  return __readfsdword(Offset);
}

// CHECK-I386-LABEL: define i32 @test__readfsdword(i32 %Offset){{.*}}{
// CHECK-I386:   [[PTR:%[0-9]+]] = inttoptr i32 %Offset to i32 addrspace(257)*
// CHECK-I386:   [[VALUE:%[0-9]+]] = load volatile i32, i32 addrspace(257)* [[PTR]], align 4
// CHECK-I386:   ret i32 [[VALUE:%[0-9]+]]
// CHECK-I386: }
#endif

__int64 test__emul(int a, int b) {
  return __emul(a, b);
}
// CHECK-LABEL: define i64 @test__emul(i32 %a, i32 %b)
// CHECK: [[X:%[0-9]+]] = sext i32 %a to i64
// CHECK: [[Y:%[0-9]+]] = sext i32 %b to i64
// CHECK: [[RES:%[0-9]+]] = mul nsw i64 [[Y]], [[X]]
// CHECK: ret i64 [[RES]]

unsigned __int64 test__emulu(unsigned int a, unsigned int b) {
  return __emulu(a, b);
}
// CHECK-LABEL: define i64 @test__emulu(i32 %a, i32 %b)
// CHECK: [[X:%[0-9]+]] = zext i32 %a to i64
// CHECK: [[Y:%[0-9]+]] = zext i32 %b to i64
// CHECK: [[RES:%[0-9]+]] = mul nuw i64 [[Y]], [[X]]
// CHECK: ret i64 [[RES]]

#if defined(__x86_64__)
__int64 test__mulh(__int64 a, __int64 b) {
  return __mulh(a, b);
}
// CHECK-X64-LABEL: define i64 @test__mulh(i64 %a, i64 %b)
// CHECK-X64: = mul nsw i128 %

unsigned __int64 test__umulh(unsigned __int64 a, unsigned __int64 b) {
  return __umulh(a, b);
}
// CHECK-X64-LABEL: define i64 @test__umulh(i64 %a, i64 %b)
// CHECK-X64: = mul nuw i128 %

__int64 test_mul128(__int64 Multiplier,
                    __int64 Multiplicand,
                    __int64 *HighProduct) {
  return _mul128(Multiplier, Multiplicand, HighProduct);
}
// CHECK-X64-LABEL: define i64 @test_mul128(i64 %Multiplier, i64 %Multiplicand, i64*{{[a-z_ ]*}}%HighProduct)
// CHECK-X64: = sext i64 %Multiplier to i128
// CHECK-X64: = sext i64 %Multiplicand to i128
// CHECK-X64: = mul nsw i128 %
// CHECK-X64: store i64 %
// CHECK-X64: ret i64 %

unsigned __int64 test_umul128(unsigned __int64 Multiplier,
                              unsigned __int64 Multiplicand,
                              unsigned __int64 *HighProduct) {
  return _umul128(Multiplier, Multiplicand, HighProduct);
}
// CHECK-X64-LABEL: define i64 @test_umul128(i64 %Multiplier, i64 %Multiplicand, i64*{{[a-z_ ]*}}%HighProduct)
// CHECK-X64: = zext i64 %Multiplier to i128
// CHECK-X64: = zext i64 %Multiplicand to i128
// CHECK-X64: = mul nuw i128 %
// CHECK-X64: store i64 %
// CHECK-X64: ret i64 %
#endif
