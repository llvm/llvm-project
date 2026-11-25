! Check SystemZ extension attributes on small BIND(C) struct arguments.

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! Only the -S run reaches the SystemZ ABI verifier, which also checks calls
! and returns with no CHECK line (assertions builds only).
! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -S %s -o /dev/null

! CHECK-LABEL: define void @s_(
! CHECK: call void @c32(i32 noext %{{.*}})
! CHECK: call void @c16(i16 noext %{{.*}})
! CHECK: call void @c8(i8 noext %{{.*}})
! CHECK: call void @c64(i64 %{{.*}})
! CHECK: call void @cf(float %{{.*}})
! CHECK: call void @c3(ptr{{.*}})
! CHECK: call void @c16b(ptr{{.*}})
! CHECK-DAG: declare void @c32(i32 noext{{.*}})
! CHECK-DAG: declare void @c16(i16 noext{{.*}})
! CHECK-DAG: declare void @c8(i8 noext{{.*}})
! CHECK-DAG: declare void @c64(i64{{.*}})
subroutine s(x32, x16, x8, x64, xf, x3, x16b)
  use iso_c_binding
  type, bind(c) :: t32
    integer(c_int32_t) :: i
  end type
  type, bind(c) :: t16
    integer(c_int16_t) :: i
  end type
  type, bind(c) :: t8
    integer(c_int8_t) :: i
  end type
  type, bind(c) :: t64
    integer(c_int64_t) :: i
  end type
  type, bind(c) :: tf
    real(c_float) :: r
  end type
  type, bind(c) :: t3
    integer(c_int8_t) :: a(3)
  end type
  type, bind(c) :: t16b
    integer(c_int32_t) :: i
    real(c_double) :: r
  end type
  interface
    subroutine c32(a) bind(c, name="c32")
      import t32
      type(t32), value :: a
    end subroutine
    subroutine c16(a) bind(c, name="c16")
      import t16
      type(t16), value :: a
    end subroutine
    subroutine c8(a) bind(c, name="c8")
      import t8
      type(t8), value :: a
    end subroutine
    subroutine c64(a) bind(c, name="c64")
      import t64
      type(t64), value :: a
    end subroutine
    subroutine cf(a) bind(c, name="cf")
      import tf
      type(tf), value :: a
    end subroutine
    subroutine c3(a) bind(c, name="c3")
      import t3
      type(t3), value :: a
    end subroutine
    subroutine c16b(a) bind(c, name="c16b")
      import t16b
      type(t16b), value :: a
    end subroutine
  end interface
  type(t32) :: x32
  type(t16) :: x16
  type(t8) :: x8
  type(t64) :: x64
  type(tf) :: xf
  type(t3) :: x3
  type(t16b) :: x16b
  call c32(x32)
  call c16(x16)
  call c8(x8)
  call c64(x64)
  call cf(xf)
  call c3(x3)
  call c16b(x16b)
end subroutine
