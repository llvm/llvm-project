!RUN: %python %S/test_errors.py %s %flang_fc1
program main
  type base
    integer :: x = 1
  end type
  type, extends(base) :: child
    integer :: y = 2
  end type
  class(child), allocatable :: c1(:), c2(:,:)
  class(base), allocatable :: b1(:), b2(:,:)
  logical var(1)
  common /blk/ var
  allocate(c1(2), c2(2,2), b1(2), b2(2,2))
  !ERROR: Actual argument for 'pad=' has type 'CLASS(base)', but 'source=' has type 'CLASS(child)'
  c2 = reshape(c1, shape(c2), pad=b1)
  b2 = reshape(b1, shape(b2), pad=c1) ! ok
  !ERROR: Actual argument for 'to=' has type 'CLASS(child)', but 'from=' has type 'CLASS(base)'
  call move_alloc(b1, c1)
  call move_alloc(c1, b1) ! ok
  !ERROR: Actual argument for 'boundary=' has type 'CLASS(base)', but 'array=' has type 'CLASS(child)'
  c1 = eoshift(c1, 1, b1(1))
  c1 = eoshift(c1, 1, c2(1,1)) ! ok
  b1 = eoshift(b1, 1, c1(1)) ! ok
  !ERROR: Actual argument for 'fsource=' has type 'CLASS(child)', but 'tsource=' has type 'CLASS(base)'
  b1 = merge(b1, c1, var(1))
  !ERROR: Actual argument for 'fsource=' has type 'CLASS(base)', but 'tsource=' has type 'CLASS(child)'
  b1 = merge(c1, b1, var(1))
  b1 = merge(b1, b1, var(1)) ! ok
  !ERROR: Actual argument for 'vector=' has type 'CLASS(base)', but 'array=' has type 'CLASS(child)'
  c1 = pack(c1, var, b1)
  c1 = pack(c1, var, c1) ! ok
  b1 = pack(b1, var, c1) ! ok
end
