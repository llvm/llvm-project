! Tests that -finit-local= with non-zero patterns produces correct LLVM IR for
! static arrays and arrays inside derived types. Previously, fir.insert_on_range
! with a non-zero element failed at LLVM lowering because llvm.mlir.constant
! does not accept ArrayAttr of non-zero scalars. The fix uses a do_loop +
! coordinate_of instead, which lowers correctly through to LLVM IR.
!
! RUN: %flang_fc1 -emit-llvm -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX %s
! RUN: %flang_fc1 -emit-llvm -finit-local=nan  %s -o - | FileCheck --check-prefix=NAN %s
! RUN: %flang_fc1 -emit-llvm -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s

! ---------------------------------------------------------------------------
! Static 1-D array INTEGER(4)(4)
! ---------------------------------------------------------------------------
subroutine test_int_array(res)
  integer(4) :: res(4)
  integer(4) :: x(4)
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_int_array{{.*}}(
! HEX:  store i32 -1431655766,
! HEX-NOT: store i32 0,

! NAN-LABEL: define {{.*}}@{{.*}}test_int_array{{.*}}(
! NAN:  store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_int_array{{.*}}(
! ZERO: store i32 0,

! ---------------------------------------------------------------------------
! Derived type with an array-valued field  (Thread 2 regression)
! type t; integer :: a(2); end type; type(t) :: x
! ---------------------------------------------------------------------------
subroutine test_array_in_struct(res)
  type :: t
    integer(4) :: a(2)
  end type
  type(t) :: res
  type(t) :: x
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_array_in_struct{{.*}}(
! HEX:  store i32 -1431655766,

! NAN-LABEL: define {{.*}}@{{.*}}test_array_in_struct{{.*}}(
! NAN:  store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_array_in_struct{{.*}}(
! ZERO: store {{.*}} zeroinitializer,

! ---------------------------------------------------------------------------
! Rank-2 array INTEGER(4)(3,4) -- flat loop must index via rank-1 view
! ---------------------------------------------------------------------------
subroutine test_int_array_2d(res)
  integer(4) :: res(3,4)
  integer(4) :: x(3,4)
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_int_array_2d{{.*}}(
! HEX:  store i32 -1431655766,
! HEX-NOT: store i32 0,

! NAN-LABEL: define {{.*}}@{{.*}}test_int_array_2d{{.*}}(
! NAN:  store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_int_array_2d{{.*}}(
! ZERO: store i32 0,

! ---------------------------------------------------------------------------
! Rank-3 array INTEGER(4)(2,3,4) -- flat loop must index via rank-1 view
! ---------------------------------------------------------------------------
subroutine test_int_array_3d(res)
  integer(4) :: res(2,3,4)
  integer(4) :: x(2,3,4)
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_int_array_3d{{.*}}(
! HEX:  store i32 -1431655766,
! HEX-NOT: store i32 0,

! NAN-LABEL: define {{.*}}@{{.*}}test_int_array_3d{{.*}}(
! NAN:  store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_int_array_3d{{.*}}(
! ZERO: store i32 0,

! ---------------------------------------------------------------------------
! Array of derived type  type(t) :: x(2)  (Thread 3/4 regression)
! Each element is a record; the loop must call initAddr per element so
! record fields are walked rather than emitting zeroinitializer.
! ---------------------------------------------------------------------------
subroutine test_array_of_struct(res)
  type :: t
    integer(4) :: a
    integer(4) :: b
  end type
  type(t) :: res(2)
  type(t) :: x(2)
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_array_of_struct{{.*}}(
! HEX:  store i32 -1431655766,
! HEX-NOT: store {{.*}} zeroinitializer,

! NAN-LABEL: define {{.*}}@{{.*}}test_array_of_struct{{.*}}(
! NAN:  store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_array_of_struct{{.*}}(
! ZERO: store {{.*}} zeroinitializer,

! ---------------------------------------------------------------------------
! Rank-2 array of derived type  type(t) :: x(2,3)
! Flat loop must stride by sizeof(%t) via rank-1 view; initAddr recurses
! into the record so all fields of all 6 elements receive the pattern.
! ---------------------------------------------------------------------------
subroutine test_array_of_struct_2d(res)
  type :: t
    integer(4) :: a
    integer(4) :: b
  end type
  type(t) :: res(2,3)
  type(t) :: x(2,3)
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_array_of_struct_2d{{.*}}(
! HEX:  store i32 -1431655766,
! HEX-NOT: store {{.*}} zeroinitializer,

! NAN-LABEL: define {{.*}}@{{.*}}test_array_of_struct_2d{{.*}}(
! NAN:  store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_array_of_struct_2d{{.*}}(
! ZERO: store {{.*}} zeroinitializer,
