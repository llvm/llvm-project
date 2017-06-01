// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple thumbv7--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686--linux -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64--linux -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG

// rotate left

unsigned char test_rotl8(unsigned char value, unsigned char shift) {
  return _rotl8(value, shift);
}
// CHECK: i8 @test_rotl8
// CHECK:   [[SHIFT:%[0-9]+]] = and i8 %{{[0-9]+}}, 7
// CHECK:   [[NEGSHIFT:%[0-9]+]] = sub i8 8, [[SHIFT]]
// CHECK:   [[HIGH:%[0-9]+]] = shl i8 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK:   [[LOW:%[0-9]+]] = lshr i8 [[VALUE]], [[NEGSHIFT]]
// CHECK:   [[ROTATED:%[0-9]+]] = or i8 [[HIGH]], [[LOW]]
// CHECK:   [[ISZERO:%[0-9]+]] = icmp eq i8 [[SHIFT]], 0
// CHECK:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i8 [[VALUE]], i8 [[ROTATED]]
// CHECK:   ret i8 [[RESULT]]
// CHECK  }

unsigned short test_rotl16(unsigned short value, unsigned char shift) {
  return _rotl16(value, shift);
}
// CHECK: i16 @test_rotl16
// CHECK:   [[SHIFT:%[0-9]+]] = and i16 %{{[0-9]+}}, 15
// CHECK:   [[NEGSHIFT:%[0-9]+]] = sub i16 16, [[SHIFT]]
// CHECK:   [[HIGH:%[0-9]+]] = shl i16 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK:   [[LOW:%[0-9]+]] = lshr i16 [[VALUE]], [[NEGSHIFT]]
// CHECK:   [[ROTATED:%[0-9]+]] = or i16 [[HIGH]], [[LOW]]
// CHECK:   [[ISZERO:%[0-9]+]] = icmp eq i16 [[SHIFT]], 0
// CHECK:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i16 [[VALUE]], i16 [[ROTATED]]
// CHECK:   ret i16 [[RESULT]]
// CHECK  }

unsigned int test_rotl(unsigned int value, int shift) {
  return _rotl(value, shift);
}
// CHECK: i32 @test_rotl
// CHECK:   [[SHIFT:%[0-9]+]] = and i32 %{{[0-9]+}}, 31
// CHECK:   [[NEGSHIFT:%[0-9]+]] = sub i32 32, [[SHIFT]]
// CHECK:   [[HIGH:%[0-9]+]] = shl i32 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK:   [[LOW:%[0-9]+]] = lshr i32 [[VALUE]], [[NEGSHIFT]]
// CHECK:   [[ROTATED:%[0-9]+]] = or i32 [[HIGH]], [[LOW]]
// CHECK:   [[ISZERO:%[0-9]+]] = icmp eq i32 [[SHIFT]], 0
// CHECK:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i32 [[VALUE]], i32 [[ROTATED]]
// CHECK:   ret i32 [[RESULT]]
// CHECK  }

unsigned long test_lrotl(unsigned long value, int shift) {
  return _lrotl(value, shift);
}
// CHECK-32BIT-LONG: i32 @test_lrotl
// CHECK-32BIT-LONG:   [[SHIFT:%[0-9]+]] = and i32 %{{[0-9]+}}, 31
// CHECK-32BIT-LONG:   [[NEGSHIFT:%[0-9]+]] = sub i32 32, [[SHIFT]]
// CHECK-32BIT-LONG:   [[HIGH:%[0-9]+]] = shl i32 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK-32BIT-LONG:   [[LOW:%[0-9]+]] = lshr i32 [[VALUE]], [[NEGSHIFT]]
// CHECK-32BIT-LONG:   [[ROTATED:%[0-9]+]] = or i32 [[HIGH]], [[LOW]]
// CHECK-32BIT-LONG:   [[ISZERO:%[0-9]+]] = icmp eq i32 [[SHIFT]], 0
// CHECK-32BIT-LONG:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i32 [[VALUE]], i32 [[ROTATED]]
// CHECK-32BIT-LONG:   ret i32 [[RESULT]]
// CHECK-32BIT-LONG  }

// CHECK-64BIT-LONG: i64 @test_lrotl
// CHECK-64BIT-LONG:   [[SHIFT:%[0-9]+]] = and i64 %{{[0-9]+}}, 63
// CHECK-64BIT-LONG:   [[NEGSHIFT:%[0-9]+]] = sub i64 64, [[SHIFT]]
// CHECK-64BIT-LONG:   [[HIGH:%[0-9]+]] = shl i64 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK-64BIT-LONG:   [[LOW:%[0-9]+]] = lshr i64 [[VALUE]], [[NEGSHIFT]]
// CHECK-64BIT-LONG:   [[ROTATED:%[0-9]+]] = or i64 [[HIGH]], [[LOW]]
// CHECK-64BIT-LONG:   [[ISZERO:%[0-9]+]] = icmp eq i64 [[SHIFT]], 0
// CHECK-64BIT-LONG:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i64 [[VALUE]], i64 [[ROTATED]]
// CHECK-64BIT-LONG:   ret i64 [[RESULT]]
// CHECK-64BIT-LONG  }

unsigned __int64 test_rotl64(unsigned __int64 value, int shift) {
  return _rotl64(value, shift);
}
// CHECK: i64 @test_rotl64
// CHECK:   [[SHIFT:%[0-9]+]] = and i64 %{{[0-9]+}}, 63
// CHECK:   [[NEGSHIFT:%[0-9]+]] = sub i64 64, [[SHIFT]]
// CHECK:   [[HIGH:%[0-9]+]] = shl i64 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK:   [[LOW:%[0-9]+]] = lshr i64 [[VALUE]], [[NEGSHIFT]]
// CHECK:   [[ROTATED:%[0-9]+]] = or i64 [[HIGH]], [[LOW]]
// CHECK:   [[ISZERO:%[0-9]+]] = icmp eq i64 [[SHIFT]], 0
// CHECK:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i64 [[VALUE]], i64 [[ROTATED]]
// CHECK:   ret i64 [[RESULT]]
// CHECK  }

// rotate right

unsigned char test_rotr8(unsigned char value, unsigned char shift) {
  return _rotr8(value, shift);
}
// CHECK: i8 @test_rotr8
// CHECK:   [[SHIFT:%[0-9]+]] = and i8 %{{[0-9]+}}, 7
// CHECK:   [[NEGSHIFT:%[0-9]+]] = sub i8 8, [[SHIFT]]
// CHECK:   [[LOW:%[0-9]+]] = lshr i8 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK:   [[HIGH:%[0-9]+]] = shl i8 [[VALUE]], [[NEGSHIFT]]
// CHECK:   [[ROTATED:%[0-9]+]] = or i8 [[HIGH]], [[LOW]]
// CHECK:   [[ISZERO:%[0-9]+]] = icmp eq i8 [[SHIFT]], 0
// CHECK:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i8 [[VALUE]], i8 [[ROTATED]]
// CHECK:   ret i8 [[RESULT]]
// CHECK  }

unsigned short test_rotr16(unsigned short value, unsigned char shift) {
  return _rotr16(value, shift);
}
// CHECK: i16 @test_rotr16
// CHECK:   [[SHIFT:%[0-9]+]] = and i16 %{{[0-9]+}}, 15
// CHECK:   [[NEGSHIFT:%[0-9]+]] = sub i16 16, [[SHIFT]]
// CHECK:   [[LOW:%[0-9]+]] = lshr i16 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK:   [[HIGH:%[0-9]+]] = shl i16 [[VALUE]], [[NEGSHIFT]]
// CHECK:   [[ROTATED:%[0-9]+]] = or i16 [[HIGH]], [[LOW]]
// CHECK:   [[ISZERO:%[0-9]+]] = icmp eq i16 [[SHIFT]], 0
// CHECK:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i16 [[VALUE]], i16 [[ROTATED]]
// CHECK:   ret i16 [[RESULT]]
// CHECK  }

unsigned int test_rotr(unsigned int value, int shift) {
  return _rotr(value, shift);
}
// CHECK: i32 @test_rotr
// CHECK:   [[SHIFT:%[0-9]+]] = and i32 %{{[0-9]+}}, 31
// CHECK:   [[NEGSHIFT:%[0-9]+]] = sub i32 32, [[SHIFT]]
// CHECK:   [[LOW:%[0-9]+]] = lshr i32 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK:   [[HIGH:%[0-9]+]] = shl i32 [[VALUE]], [[NEGSHIFT]]
// CHECK:   [[ROTATED:%[0-9]+]] = or i32 [[HIGH]], [[LOW]]
// CHECK:   [[ISZERO:%[0-9]+]] = icmp eq i32 [[SHIFT]], 0
// CHECK:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i32 [[VALUE]], i32 [[ROTATED]]
// CHECK:   ret i32 [[RESULT]]
// CHECK  }

unsigned long test_lrotr(unsigned long value, int shift) {
  return _lrotr(value, shift);
}
// CHECK-32BIT-LONG: i32 @test_lrotr
// CHECK-32BIT-LONG:   [[SHIFT:%[0-9]+]] = and i32 %{{[0-9]+}}, 31
// CHECK-32BIT-LONG:   [[NEGSHIFT:%[0-9]+]] = sub i32 32, [[SHIFT]]
// CHECK-32BIT-LONG:   [[LOW:%[0-9]+]] = lshr i32 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK-32BIT-LONG:   [[HIGH:%[0-9]+]] = shl i32 [[VALUE]], [[NEGSHIFT]]
// CHECK-32BIT-LONG:   [[ROTATED:%[0-9]+]] = or i32 [[HIGH]], [[LOW]]
// CHECK-32BIT-LONG:   [[ISZERO:%[0-9]+]] = icmp eq i32 [[SHIFT]], 0
// CHECK-32BIT-LONG:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i32 [[VALUE]], i32 [[ROTATED]]
// CHECK-32BIT-LONG:   ret i32 [[RESULT]]
// CHECK-32BIT-LONG  }

// CHECK-64BIT-LONG: i64 @test_lrotr
// CHECK-64BIT-LONG:   [[SHIFT:%[0-9]+]] = and i64 %{{[0-9]+}}, 63
// CHECK-64BIT-LONG:   [[NEGSHIFT:%[0-9]+]] = sub i64 64, [[SHIFT]]
// CHECK-64BIT-LONG:   [[LOW:%[0-9]+]] = lshr i64 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK-64BIT-LONG:   [[HIGH:%[0-9]+]] = shl i64 [[VALUE]], [[NEGSHIFT]]
// CHECK-64BIT-LONG:   [[ROTATED:%[0-9]+]] = or i64 [[HIGH]], [[LOW]]
// CHECK-64BIT-LONG:   [[ISZERO:%[0-9]+]] = icmp eq i64 [[SHIFT]], 0
// CHECK-64BIT-LONG:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i64 [[VALUE]], i64 [[ROTATED]]
// CHECK-64BIT-LONG:   ret i64 [[RESULT]]
// CHECK-64BIT-LONG  }

unsigned __int64 test_rotr64(unsigned __int64 value, int shift) {
  return _rotr64(value, shift);
}
// CHECK: i64 @test_rotr64
// CHECK:   [[SHIFT:%[0-9]+]] = and i64 %{{[0-9]+}}, 63
// CHECK:   [[NEGSHIFT:%[0-9]+]] = sub i64 64, [[SHIFT]]
// CHECK:   [[LOW:%[0-9]+]] = lshr i64 [[VALUE:%[0-9]+]], [[SHIFT]]
// CHECK:   [[HIGH:%[0-9]+]] = shl i64 [[VALUE]], [[NEGSHIFT]]
// CHECK:   [[ROTATED:%[0-9]+]] = or i64 [[HIGH]], [[LOW]]
// CHECK:   [[ISZERO:%[0-9]+]] = icmp eq i64 [[SHIFT]], 0
// CHECK:   [[RESULT:%[0-9]+]] = select i1 [[ISZERO]], i64 [[VALUE]], i64 [[ROTATED]]
// CHECK:   ret i64 [[RESULT]]
// CHECK  }
