! Tests that -finit-local= with non-zero patterns produces correct LLVM IR for
! static arrays and arrays inside derived types. Previously, fir.insert_on_range
! with a non-zero element failed at LLVM lowering because llvm.mlir.constant
! does not accept ArrayAttr of non-zero scalars. The fix uses a do_loop +
! coordinate_of instead, which lowers correctly through to LLVM IR.
!
! The HEX checks verify:
!   - the loop trip counter PHI starts at the expected element count,
!   - the GEP uses the element type as the unit stride (so all elements are
!     reached, not just element 0), and
!   - the store writes the expected bit pattern on every iteration.
! These three properties together prove that every element is initialized.
!
! RUN: %flang_fc1 -emit-llvm -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX %s
! RUN: %flang_fc1 -emit-llvm -finit-local=nan  %s -o - | FileCheck --check-prefix=NAN %s
! RUN: %flang_fc1 -emit-llvm -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s

! ---------------------------------------------------------------------------
! Static 1-D array INTEGER(4)(4) -- 4 elements
! ---------------------------------------------------------------------------
subroutine test_int_array(res)
  integer(4) :: res(4)
  integer(4) :: x(4)
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_int_array{{.*}}(
! HEX:         phi i64 [ {{.*}}, {{.*}} ], [ 4, %{{.*}} ]
! HEX:         getelementptr i32, ptr {{.*}}, i64
! HEX:         store i32 -1431655766,
! HEX-NOT:     store i32 0,

! NAN-LABEL: define {{.*}}@{{.*}}test_int_array{{.*}}(
! NAN:         phi i64 [ {{.*}}, {{.*}} ], [ 4, %{{.*}} ]
! NAN:         getelementptr i32, ptr {{.*}}, i64
! NAN:         store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_int_array{{.*}}(
! ZERO:        phi i64 [ {{.*}}, {{.*}} ], [ 4, %{{.*}} ]
! ZERO:        getelementptr i32, ptr {{.*}}, i64
! ZERO:        store i32 0,

! ---------------------------------------------------------------------------
! Derived type with an array-valued field (Thread 2 regression)
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
! HEX:         phi i64 [ {{.*}}, {{.*}} ], [ 8, %{{.*}} ]
! HEX:         getelementptr i8, ptr {{.*}}, i64
! HEX:         store i8 -86,

! NAN-LABEL: define {{.*}}@{{.*}}test_array_in_struct{{.*}}(
! NAN:         phi i64 [ {{.*}}, {{.*}} ], [ 2, %{{.*}} ]
! NAN:         getelementptr i32, ptr {{.*}}, i64
! NAN:         store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_array_in_struct{{.*}}(
! ZERO:        store {{.*}} zeroinitializer,

! ---------------------------------------------------------------------------
! Rank-2 array INTEGER(4)(3,4) -- 12 elements; flat loop via rank-1 view
! ---------------------------------------------------------------------------
subroutine test_int_array_2d(res)
  integer(4) :: res(3,4)
  integer(4) :: x(3,4)
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_int_array_2d{{.*}}(
! HEX:         phi i64 [ {{.*}}, {{.*}} ], [ 12, %{{.*}} ]
! HEX:         getelementptr i32, ptr {{.*}}, i64
! HEX:         store i32 -1431655766,
! HEX-NOT:     store i32 0,

! NAN-LABEL: define {{.*}}@{{.*}}test_int_array_2d{{.*}}(
! NAN:         phi i64 [ {{.*}}, {{.*}} ], [ 12, %{{.*}} ]
! NAN:         getelementptr i32, ptr {{.*}}, i64
! NAN:         store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_int_array_2d{{.*}}(
! ZERO:        phi i64 [ {{.*}}, {{.*}} ], [ 12, %{{.*}} ]
! ZERO:        getelementptr i32, ptr {{.*}}, i64
! ZERO:        store i32 0,

! ---------------------------------------------------------------------------
! Rank-3 array INTEGER(4)(2,3,4) -- 24 elements; flat loop via rank-1 view
! ---------------------------------------------------------------------------
subroutine test_int_array_3d(res)
  integer(4) :: res(2,3,4)
  integer(4) :: x(2,3,4)
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_int_array_3d{{.*}}(
! HEX:         phi i64 [ {{.*}}, {{.*}} ], [ 24, %{{.*}} ]
! HEX:         getelementptr i32, ptr {{.*}}, i64
! HEX:         store i32 -1431655766,
! HEX-NOT:     store i32 0,

! NAN-LABEL: define {{.*}}@{{.*}}test_int_array_3d{{.*}}(
! NAN:         phi i64 [ {{.*}}, {{.*}} ], [ 24, %{{.*}} ]
! NAN:         getelementptr i32, ptr {{.*}}, i64
! NAN:         store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_int_array_3d{{.*}}(
! ZERO:        phi i64 [ {{.*}}, {{.*}} ], [ 24, %{{.*}} ]
! ZERO:        getelementptr i32, ptr {{.*}}, i64
! ZERO:        store i32 0,

! ---------------------------------------------------------------------------
! Array of derived type  type(t) :: x(2)  (Thread 3/4 regression)
! Loop strides by sizeof(%t); initAddr recurses into each record element so
! all fields receive the pattern rather than a zeroinitializer.
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
! HEX:         phi i64 [ {{.*}}, {{.*}} ], [ 2, %{{.*}} ]
! HEX:         getelementptr %{{.*}}t, ptr {{.*}}, i64
! HEX:         getelementptr i8, ptr {{.*}}, i64
! HEX:         store i8 -86,
! HEX-NOT:     store {{.*}} zeroinitializer,

! NAN-LABEL: define {{.*}}@{{.*}}test_array_of_struct{{.*}}(
! NAN:         phi i64 [ {{.*}}, {{.*}} ], [ 2, %{{.*}} ]
! NAN:         getelementptr %{{.*}}t, ptr {{.*}}, i64
! NAN:         store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_array_of_struct{{.*}}(
! ZERO:        phi i64 [ {{.*}}, {{.*}} ], [ 2, %{{.*}} ]
! ZERO:        getelementptr %{{.*}}t, ptr {{.*}}, i64
! ZERO:        store {{.*}} zeroinitializer,

! ---------------------------------------------------------------------------
! Rank-2 array of derived type  type(t) :: x(2,3) -- 6 elements
! Flat loop must stride by sizeof(%t) via rank-1 view; initAddr recurses
! into each record so all fields of all 6 elements receive the pattern.
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
! HEX:         phi i64 [ {{.*}}, {{.*}} ], [ 6, %{{.*}} ]
! HEX:         getelementptr %{{.*}}t, ptr {{.*}}, i64
! HEX:         getelementptr i8, ptr {{.*}}, i64
! HEX:         store i8 -86,
! HEX-NOT:     store {{.*}} zeroinitializer,

! NAN-LABEL: define {{.*}}@{{.*}}test_array_of_struct_2d{{.*}}(
! NAN:         phi i64 [ {{.*}}, {{.*}} ], [ 6, %{{.*}} ]
! NAN:         getelementptr %{{.*}}t, ptr {{.*}}, i64
! NAN:         store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_array_of_struct_2d{{.*}}(
! ZERO:        phi i64 [ {{.*}}, {{.*}} ], [ 6, %{{.*}} ]
! ZERO:        getelementptr %{{.*}}t, ptr {{.*}}, i64
! ZERO:        store {{.*}} zeroinitializer,
