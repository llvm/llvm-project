! RUN: %flang_fc1 -emit-obj -flang-experimental-hlfir -o /dev/null %s

! Regression test: ensure we can compile this without crashing
! this results in a hlfir.elemental with mismatched types in the hlfir.apply
! and hlfir.yield
subroutine test
  interface
    function func(i,j,k)
      character(5),allocatable :: func(:,:,:)
    end function func
  end interface
  character(13),allocatable :: a(:,:,:)
  print *, (func(2,5,3)//reshape([(char(ichar('a')+n),n=1,2*5*3)], &
    & [2,5,3]))
end subroutine test
