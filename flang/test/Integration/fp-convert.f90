! RUN: %flang -funsigned %s -o %t && %t | FileCheck %s
! RUN: %flang -funsigned -emit-llvm -S -o - %s | FileCheck %s --check-prefix=LLVMIR

module fp_convert_m
  implicit none
  interface set_and_print
    module procedure set_and_print_r16
    module procedure set_and_print_r8
  end interface
contains
  subroutine set_and_print_r16(value)
    real(kind=16), intent(in) :: value
    integer(kind=1) :: i8
    integer(kind=2) :: i16
    integer(kind=4) :: i32
    integer(kind=8) :: i64
    integer(kind=16) :: i128
    unsigned(kind=1) :: u8
    unsigned(kind=2) :: u16
    unsigned(kind=4) :: u32
    unsigned(kind=8) :: u64
    unsigned(kind=16) :: u128
    print *, "Original real(16) value:", value
    i8 = int(value, kind=1)
    i16 = int(value, kind=2)
    i32 = int(value, kind=4)
    i64 = int(value, kind=8)
    i128 = int(value, kind=16)
    u8 = uint(value, kind=1)
    u16 = uint(value, kind=2)
    u32 = uint(value, kind=4)
    u64 = uint(value, kind=8)
    u128 = uint(value, kind=16)
    print *, "Converted to 8-bit integer:", i8
    print *, "Converted to 16-bit integer:", i16
    print *, "Converted to 32-bit integer:", i32
    print *, "Converted to 64-bit integer:", i64
    print *, "Converted to 128-bit integer:", i128
    print *, "Converted to 8-bit unsigned integer:", u8
    print *, "Converted to 16-bit unsigned integer:", u16
    print *, "Converted to 32-bit unsigned integer:", u32
    print *, "Converted to 64-bit unsigned integer:", u64
    print *, "Converted to 128-bit unsigned integer:", u128
  end subroutine

  subroutine set_and_print_r8(value)
    real(kind=8), intent(in) :: value
    integer(kind=1) :: i8
    integer(kind=2) :: i16
    integer(kind=4) :: i32
    integer(kind=8) :: i64
    integer(kind=16) :: i128
    unsigned(kind=1) :: u8
    unsigned(kind=2) :: u16
    unsigned(kind=4) :: u32
    unsigned(kind=8) :: u64
    unsigned(kind=16) :: u128
    print *, "Original real(8) value:", value
    i8 = int(value, kind=1)
    i16 = int(value, kind=2)
    i32 = int(value, kind=4)
    i64 = int(value, kind=8)
    i128 = int(value, kind=16)
    u8 = uint(value, kind=1)
    u16 = uint(value, kind=2)
    u32 = uint(value, kind=4)
    u64 = uint(value, kind=8)
    u128 = uint(value, kind=16)
    print *, "Converted to 8-bit integer:", i8
    print *, "Converted to 16-bit integer:", i16
    print *, "Converted to 32-bit integer:", i32
    print *, "Converted to 64-bit integer:", i64
    print *, "Converted to 128-bit integer:", i128
    print *, "Converted to 8-bit unsigned integer:", u8
    print *, "Converted to 16-bit unsigned integer:", u16
    print *, "Converted to 32-bit unsigned integer:", u32
    print *, "Converted to 64-bit unsigned integer:", u64
    print *, "Converted to 128-bit unsigned integer:", u128
  end subroutine
end module fp_convert_m

program fp_convert
  use ieee_arithmetic, only: ieee_value, ieee_quiet_nan, ieee_positive_inf, ieee_negative_inf
  use fp_convert_m, only: set_and_print
  implicit none

  real(kind=8) :: nan, inf, ninf
  nan = ieee_value(nan, ieee_quiet_nan)
  inf = ieee_value(inf, ieee_positive_inf)
  ninf = ieee_value(ninf, ieee_negative_inf)

  call set_and_print(huge(0.0_8))
  call set_and_print(-huge(0.0_8))
  call set_and_print(huge(0.0_16))
  call set_and_print(-huge(0.0_16))
  call set_and_print(tiny(0.0_8))
  call set_and_print(-tiny(0.0_8))
  call set_and_print(tiny(0.0_16))
  call set_and_print(-tiny(0.0_16))
  call set_and_print(nan)
  call set_and_print(inf)
  call set_and_print(ninf)

end program fp_convert

! LLVMIR: call i8 @llvm.fptosi.sat.i8.f128(fp128 %{{.+}})
! LLVMIR: call i16 @llvm.fptosi.sat.i16.f128(fp128 %{{.+}})
! LLVMIR: call i32 @llvm.fptosi.sat.i32.f128(fp128 %{{.+}})
! LLVMIR: call i64 @llvm.fptosi.sat.i64.f128(fp128 %{{.+}})
! LLVMIR: call i128 @llvm.fptosi.sat.i128.f128(fp128 %{{.+}})
! LLVMIR: call i8 @llvm.fptoui.sat.i8.f128(fp128 %{{.+}})
! LLVMIR: call i16 @llvm.fptoui.sat.i16.f128(fp128 %{{.+}})
! LLVMIR: call i32 @llvm.fptoui.sat.i32.f128(fp128 %{{.+}})
! LLVMIR: call i64 @llvm.fptoui.sat.i64.f128(fp128 %{{.+}})
! LLVMIR: call i128 @llvm.fptoui.sat.i128.f128(fp128 %{{.+}})
! LLVMIR: call i8 @llvm.fptosi.sat.i8.f64(double %{{.+}})
! LLVMIR: call i16 @llvm.fptosi.sat.i16.f64(double %{{.+}})
! LLVMIR: call i32 @llvm.fptosi.sat.i32.f64(double %{{.+}})
! LLVMIR: call i64 @llvm.fptosi.sat.i64.f64(double %{{.+}})
! LLVMIR: call i128 @llvm.fptosi.sat.i128.f64(double %{{.+}})
! LLVMIR: call i8 @llvm.fptoui.sat.i8.f64(double %{{.+}})
! LLVMIR: call i16 @llvm.fptoui.sat.i16.f64(double %{{.+}})
! LLVMIR: call i32 @llvm.fptoui.sat.i32.f64(double %{{.+}})
! LLVMIR: call i64 @llvm.fptoui.sat.i64.f64(double %{{.+}})
! LLVMIR: call i128 @llvm.fptoui.sat.i128.f64(double %{{.+}})

! CHECK: Converted to 8-bit integer: 127
! CHECK: Converted to 16-bit integer: 32767
! CHECK: Converted to 32-bit integer: 2147483647
! CHECK: Converted to 64-bit integer: 9223372036854775807
! CHECK: Converted to 128-bit integer: 170141183460469231731687303715884105727
! CHECK: Converted to 8-bit unsigned integer: 255
! CHECK: Converted to 16-bit unsigned integer: 65535
! CHECK: Converted to 32-bit unsigned integer: 4294967295
! CHECK: Converted to 64-bit unsigned integer: 18446744073709551615
! CHECK: Converted to 128-bit unsigned integer: 340282366920938463463374607431768211455
! CHECK: Converted to 8-bit integer: -128
! CHECK: Converted to 16-bit integer: -32768
! CHECK: Converted to 32-bit integer: -2147483648
! CHECK: Converted to 64-bit integer: -9223372036854775808
! CHECK: Converted to 128-bit integer: -170141183460469231731687303715884105728
! CHECK: Converted to 8-bit unsigned integer: 0
! CHECK: Converted to 16-bit unsigned integer: 0
! CHECK: Converted to 32-bit unsigned integer: 0
! CHECK: Converted to 64-bit unsigned integer: 0
! CHECK: Converted to 128-bit unsigned integer: 0
! CHECK: Converted to 8-bit integer: 127
! CHECK: Converted to 16-bit integer: 32767
! CHECK: Converted to 32-bit integer: 2147483647
! CHECK: Converted to 64-bit integer: 9223372036854775807
! CHECK: Converted to 128-bit integer: 170141183460469231731687303715884105727
! CHECK: Converted to 8-bit unsigned integer: 255
! CHECK: Converted to 16-bit unsigned integer: 65535
! CHECK: Converted to 32-bit unsigned integer: 4294967295
! CHECK: Converted to 64-bit unsigned integer: 18446744073709551615
! CHECK: Converted to 128-bit unsigned integer: 340282366920938463463374607431768211455
! CHECK: Converted to 8-bit integer: -128
! CHECK: Converted to 16-bit integer: -32768
! CHECK: Converted to 32-bit integer: -2147483648
! CHECK: Converted to 64-bit integer: -9223372036854775808
! CHECK: Converted to 128-bit integer: -170141183460469231731687303715884105728
! CHECK: Converted to 8-bit unsigned integer: 0
! CHECK: Converted to 16-bit unsigned integer: 0
! CHECK: Converted to 32-bit unsigned integer: 0
! CHECK: Converted to 64-bit unsigned integer: 0
! CHECK: Converted to 128-bit unsigned integer: 0
! CHECK: Converted to 8-bit integer: 0
! CHECK: Converted to 16-bit integer: 0
! CHECK: Converted to 32-bit integer: 0
! CHECK: Converted to 64-bit integer: 0
! CHECK: Converted to 128-bit integer: 0
! CHECK: Converted to 8-bit unsigned integer: 0
! CHECK: Converted to 16-bit unsigned integer: 0
! CHECK: Converted to 32-bit unsigned integer: 0
! CHECK: Converted to 64-bit unsigned integer: 0
! CHECK: Converted to 128-bit unsigned integer: 0
! CHECK: Converted to 8-bit integer: 0
! CHECK: Converted to 16-bit integer: 0
! CHECK: Converted to 32-bit integer: 0
! CHECK: Converted to 64-bit integer: 0
! CHECK: Converted to 128-bit integer: 0
! CHECK: Converted to 8-bit unsigned integer: 0
! CHECK: Converted to 16-bit unsigned integer: 0
! CHECK: Converted to 32-bit unsigned integer: 0
! CHECK: Converted to 64-bit unsigned integer: 0
! CHECK: Converted to 128-bit unsigned integer: 0
! CHECK: Converted to 8-bit integer: 0
! CHECK: Converted to 16-bit integer: 0
! CHECK: Converted to 32-bit integer: 0
! CHECK: Converted to 64-bit integer: 0
! CHECK: Converted to 128-bit integer: 0
! CHECK: Converted to 8-bit unsigned integer: 0
! CHECK: Converted to 16-bit unsigned integer: 0
! CHECK: Converted to 32-bit unsigned integer: 0
! CHECK: Converted to 64-bit unsigned integer: 0
! CHECK: Converted to 128-bit unsigned integer: 0
! CHECK: Converted to 8-bit integer: 0
! CHECK: Converted to 16-bit integer: 0
! CHECK: Converted to 32-bit integer: 0
! CHECK: Converted to 64-bit integer: 0
! CHECK: Converted to 128-bit integer: 0
! CHECK: Converted to 8-bit unsigned integer: 0
! CHECK: Converted to 16-bit unsigned integer: 0
! CHECK: Converted to 32-bit unsigned integer: 0
! CHECK: Converted to 64-bit unsigned integer: 0
! CHECK: Converted to 128-bit unsigned integer: 0
! CHECK: Converted to 8-bit integer: 0
! CHECK: Converted to 16-bit integer: 0
! CHECK: Converted to 32-bit integer: 0
! CHECK: Converted to 64-bit integer: 0
! CHECK: Converted to 128-bit integer: 0
! CHECK: Converted to 8-bit unsigned integer: 0
! CHECK: Converted to 16-bit unsigned integer: 0
! CHECK: Converted to 32-bit unsigned integer: 0
! CHECK: Converted to 64-bit unsigned integer: 0
! CHECK: Converted to 128-bit unsigned integer: 0
! CHECK: Converted to 8-bit integer: 127
! CHECK: Converted to 16-bit integer: 32767
! CHECK: Converted to 32-bit integer: 2147483647
! CHECK: Converted to 64-bit integer: 9223372036854775807
! CHECK: Converted to 128-bit integer: 170141183460469231731687303715884105727
! CHECK: Converted to 8-bit unsigned integer: 255
! CHECK: Converted to 16-bit unsigned integer: 65535
! CHECK: Converted to 32-bit unsigned integer: 4294967295
! CHECK: Converted to 64-bit unsigned integer: 18446744073709551615
! CHECK: Converted to 128-bit unsigned integer: 340282366920938463463374607431768211455
! CHECK: Converted to 8-bit integer: -128
! CHECK: Converted to 16-bit integer: -32768
! CHECK: Converted to 32-bit integer: -2147483648
! CHECK: Converted to 64-bit integer: -9223372036854775808
! CHECK: Converted to 128-bit integer: -170141183460469231731687303715884105728
! CHECK: Converted to 8-bit unsigned integer: 0
! CHECK: Converted to 16-bit unsigned integer: 0
! CHECK: Converted to 32-bit unsigned integer: 0
! CHECK: Converted to 64-bit unsigned integer: 0
! CHECK: Converted to 128-bit unsigned integer: 0
