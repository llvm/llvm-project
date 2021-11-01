!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

! Size and alignment of intrinsic types
subroutine s1
  integer(1) :: a_i1  !CHECK: a_i1 size=1 offset=0:
  integer(8) :: b_i8  !CHECK: b_i8 size=8 offset=8:
  real(2)    :: c_r2  !CHECK: c_r2 size=2 offset=16:
  real(2)    :: d_r2  !CHECK: d_r2 size=2 offset=18:
  real(8)    :: e_r8  !CHECK: e_r8 size=8 offset=24:
  real(4)    :: f_r4  !CHECK: f_r4 size=4 offset=32:
  complex(8) :: g_c8  !CHECK: g_c8 size=16 offset=40:
  complex(4) :: h_c4  !CHECK: h_c4 size=8 offset=56:
  logical    :: i_l4  !CHECK: i_l4 size=4 offset=64:
end

! Character
subroutine s2
  character(10)        :: c1 !CHECK: c1 size=10 offset=0:
  character(1)         :: c2 !CHECK: c2 size=1 offset=10:
  character(10,kind=2) :: c3 !CHECK: c3 size=20 offset=12:
end

! Descriptors
subroutine s3(n)
  integer :: n
  real, pointer :: x !CHECK: x, POINTER size=24 offset=8:
  character(n)  :: y !CHECK: y size=24 offset=32:
end

! Descriptors for arrays
subroutine s4
  integer, allocatable :: z0        !CHECK: z0, ALLOCATABLE size=24 offset=
  integer, allocatable :: z1(:)     !CHECK: z1, ALLOCATABLE size=48 offset=
  integer, allocatable :: z2(:,:)   !CHECK: z2, ALLOCATABLE size=72 offset=
  integer, allocatable :: z3(:,:,:) !CHECK: z3, ALLOCATABLE size=96 offset=
end

! Descriptors with length parameters
subroutine s5(n)
  integer :: n
  type :: t1(n)
    integer, len :: n
    real :: a(n)
  end type
  type :: t2(n1, n2)
    integer, len :: n1
    integer, len :: n2
    real :: b(n1, n2)
  end type
  type(t1(n))   :: x1 !CHECK: x1 size=40 offset=
  type(t2(n,n)) :: x2 !CHECK: x2 size=48 offset=
  !CHECK: a size=48 offset=0:
  !CHECK: b size=72 offset=0:
end
