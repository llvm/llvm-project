! RUN: %python %S/test_modfile.py %s %flang_fc1 -fopenmp
! Check correct modfile generation for OpenMP threadprivate directive.

module m
  implicit none
  type :: my_type(kind_param, len_param)
    integer, KIND :: kind_param
    integer, LEN :: len_param
    integer :: t_i
    integer :: t_arr(10)
  end type
  type(my_type(kind_param=2, len_param=4)) :: t
  real, dimension(3) :: thrtest
  real :: x
  common /blk/ x

  !$omp threadprivate(thrtest, t, /blk/)
end

!Expect: m.mod
!module m
!type::my_type(kind_param,len_param)
!integer(4),kind::kind_param
!integer(4),len::len_param
!integer(4)::t_i
!integer(4)::t_arr(1_8:10_8)
!end type
!type(my_type(kind_param=2_4,len_param=4_4))::t
!!$omp threadprivate(t)
!real(4)::thrtest(1_8:3_8)
!!$omp threadprivate(thrtest)
!real(4)::x
!!$omp threadprivate(x)
!common/blk/x
!end
