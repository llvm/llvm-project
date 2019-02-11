// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-I386
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-X64

#if defined(__i386__)
char test__readfsbyte(unsigned long Offset) {
  return __readfsbyte(++Offset);
}
// CHECK-I386-LABEL: define dso_local signext i8 @test__readfsbyte(i32 %Offset)
// CHECK-I386:   %inc = add i32 %Offset, 1
// CHECK-I386:   [[PTR:%[0-9]+]] = inttoptr i32 %inc to i8 addrspace(257)*
// CHECK-I386:   [[VALUE:%[0-9]+]] = load volatile i8, i8 addrspace(257)* [[PTR]], align 1
// CHECK-I386:   ret i8 [[VALUE:%[0-9]+]]

short test__readfsword(unsigned long Offset) {
  return __readfsword(++Offset);
}
// CHECK-I386-LABEL: define dso_local signext i16 @test__readfsword(i32 %Offset)
// CHECK-I386:   %inc = add i32 %Offset, 1
// CHECK-I386:   [[PTR:%[0-9]+]] = inttoptr i32 %inc to i16 addrspace(257)*
// CHECK-I386:   [[VALUE:%[0-9]+]] = load volatile i16, i16 addrspace(257)* [[PTR]], align 2
// CHECK-I386:   ret i16 [[VALUE:%[0-9]+]]

long test__readfsdword(unsigned long Offset) {
  return __readfsdword(++Offset);
}
// CHECK-I386-LABEL: define dso_local i32 @test__readfsdword(i32 %Offset)
// CHECK-I386:   %inc = add i32 %Offset, 1
// CHECK-I386:   [[PTR:%[0-9]+]] = inttoptr i32 %inc to i32 addrspace(257)*
// CHECK-I386:   [[VALUE:%[0-9]+]] = load volatile i32, i32 addrspace(257)* [[PTR]], align 4
// CHECK-I386:   ret i32 [[VALUE:%[0-9]+]]

long long test__readfsqword(unsigned long Offset) {
  return __readfsqword(++Offset);
}
// CHECK-I386-LABEL: define dso_local i64 @test__readfsqword(i32 %Offset)
// CHECK-I386:   %inc = add i32 %Offset, 1
// CHECK-I386:   [[PTR:%[0-9]+]] = inttoptr i32 %inc to i64 addrspace(257)*
// CHECK-I386:   [[VALUE:%[0-9]+]] = load volatile i64, i64 addrspace(257)* [[PTR]], align 8
// CHECK-I386:   ret i64 [[VALUE:%[0-9]+]]
#endif

__int64 test__emul(int a, int b) {
  return __emul(a, b);
}
// CHECK-LABEL: define dso_local i64 @test__emul(i32 %a, i32 %b)
// CHECK: [[X:%[0-9]+]] = sext i32 %a to i64
// CHECK: [[Y:%[0-9]+]] = sext i32 %b to i64
// CHECK: [[RES:%[0-9]+]] = mul nsw i64 [[Y]], [[X]]
// CHECK: ret i64 [[RES]]

unsigned __int64 test__emulu(unsigned int a, unsigned int b) {
  return __emulu(a, b);
}
// CHECK-LABEL: define dso_local i64 @test__emulu(i32 %a, i32 %b)
// CHECK: [[X:%[0-9]+]] = zext i32 %a to i64
// CHECK: [[Y:%[0-9]+]] = zext i32 %b to i64
// CHECK: [[RES:%[0-9]+]] = mul nuw i64 [[Y]], [[X]]
// CHECK: ret i64 [[RES]]

#if defined(__x86_64__)

char test__readgsbyte(unsigned long Offset) {
  return __readgsbyte(++Offset);
}
// CHECK-X64-LABEL: define dso_local i8 @test__readgsbyte(i32 %Offset)
// CHECK-X64:   %inc = add i32 %Offset, 1
// CHECK-X64:   [[ZEXT:%[0-9]+]] = zext i32 %inc to i64
// CHECK-X64:   [[PTR:%[0-9]+]] = inttoptr i64 [[ZEXT]] to i8 addrspace(256)*
// CHECK-X64:   [[VALUE:%[0-9]+]] = load volatile i8, i8 addrspace(256)* [[PTR]], align 1
// CHECK-X64:   ret i8 [[VALUE:%[0-9]+]]

short test__readgsword(unsigned long Offset) {
  return __readgsword(++Offset);
}
// CHECK-X64-LABEL: define dso_local i16 @test__readgsword(i32 %Offset)
// CHECK-X64:   %inc = add i32 %Offset, 1
// CHECK-X64:   [[ZEXT:%[0-9]+]] = zext i32 %inc to i64
// CHECK-X64:   [[PTR:%[0-9]+]] = inttoptr i64 [[ZEXT]] to i16 addrspace(256)*
// CHECK-X64:   [[VALUE:%[0-9]+]] = load volatile i16, i16 addrspace(256)* [[PTR]], align 2
// CHECK-X64:   ret i16 [[VALUE:%[0-9]+]]

long test__readgsdword(unsigned long Offset) {
  return __readgsdword(++Offset);
}
// CHECK-X64-LABEL: define dso_local i32 @test__readgsdword(i32 %Offset)
// CHECK-X64:   %inc = add i32 %Offset, 1
// CHECK-X64:   [[ZEXT:%[0-9]+]] = zext i32 %inc to i64
// CHECK-X64:   [[PTR:%[0-9]+]] = inttoptr i64 [[ZEXT]] to i32 addrspace(256)*
// CHECK-X64:   [[VALUE:%[0-9]+]] = load volatile i32, i32 addrspace(256)* [[PTR]], align 4
// CHECK-X64:   ret i32 [[VALUE:%[0-9]+]]

long long test__readgsqword(unsigned long Offset) {
  return __readgsqword(++Offset);
}
// CHECK-X64-LABEL: define dso_local i64 @test__readgsqword(i32 %Offset)
// CHECK-X64:   %inc = add i32 %Offset, 1
// CHECK-X64:   [[ZEXT:%[0-9]+]] = zext i32 %inc to i64
// CHECK-X64:   [[PTR:%[0-9]+]] = inttoptr i64 [[ZEXT]] to i64 addrspace(256)*
// CHECK-X64:   [[VALUE:%[0-9]+]] = load volatile i64, i64 addrspace(256)* [[PTR]], align 8
// CHECK-X64:   ret i64 [[VALUE:%[0-9]+]]

__int64 test__mulh(__int64 a, __int64 b) {
  return __mulh(a, b);
}
// CHECK-X64-LABEL: define dso_local i64 @test__mulh(i64 %a, i64 %b)
// CHECK-X64: = mul nsw i128 %

unsigned __int64 test__umulh(unsigned __int64 a, unsigned __int64 b) {
  return __umulh(a, b);
}
// CHECK-X64-LABEL: define dso_local i64 @test__umulh(i64 %a, i64 %b)
// CHECK-X64: = mul nuw i128 %

__int64 test_mul128(__int64 Multiplier,
                    __int64 Multiplicand,
                    __int64 *HighProduct) {
  return _mul128(Multiplier, Multiplicand, HighProduct);
}
// CHECK-X64-LABEL: define dso_local i64 @test_mul128(i64 %Multiplier, i64 %Multiplicand, i64*{{[a-z_ ]*}}%HighProduct)
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
// CHECK-X64-LABEL: define dso_local i64 @test_umul128(i64 %Multiplier, i64 %Multiplicand, i64*{{[a-z_ ]*}}%HighProduct)
// CHECK-X64: = zext i64 %Multiplier to i128
// CHECK-X64: = zext i64 %Multiplicand to i128
// CHECK-X64: = mul nuw i128 %
// CHECK-X64: store i64 %
// CHECK-X64: ret i64 %

unsigned __int64 test__shiftleft128(unsigned __int64 l, unsigned __int64 h,
                                    unsigned char d) {
  return __shiftleft128(l, h, d);
}
// CHECK-X64-LABEL: define dso_local i64 @test__shiftleft128(i64 %l, i64 %h, i8 %d)
// CHECK-X64:  = zext i64 %{{.*}} to i128
// CHECK-X64:  = shl nuw i128 %{{.*}}, 64
// CHECK-X64:  = zext i64 %{{.*}} to i128
// CHECK-X64:  = or i128 %
// CHECK-X64:  = and i8 %{{.*}}, 63
// CHECK-X64:  = shl i128 %
// CHECK-X64:  = lshr i128 %
// CHECK-X64:  = trunc i128 %
// CHECK-X64:  ret i64 %

unsigned __int64 test__shiftright128(unsigned __int64 l, unsigned __int64 h,
                                     unsigned char d) {
  return __shiftright128(l, h, d);
}
// CHECK-X64-LABEL: define dso_local i64 @test__shiftright128(i64 %l, i64 %h, i8 %d)
// CHECK-X64:  = zext i64 %{{.*}} to i128
// CHECK-X64:  = shl nuw i128 %{{.*}}, 64
// CHECK-X64:  = zext i64 %{{.*}} to i128
// CHECK-X64:  = or i128 %
// CHECK-X64:  = and i8 %{{.*}}, 63
// CHECK-X64:  = lshr i128 %
// CHECK-X64:  = trunc i128 %
// CHECK-X64:  ret i64 %

#endif // defined(__x86_64__)
