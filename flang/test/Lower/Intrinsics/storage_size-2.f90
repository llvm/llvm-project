! Test storage_size with characters
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! check-label: func.func @_QPtest_storage_size
subroutine test_storage_size(n)
  interface
    function return_char(l)
      integer :: l
      character(l) :: return_char
    end function
  end interface
  integer n
  print*, storage_size(return_char(n))
! CHECK: %[[val_16:.*]] = fir.call @_QPreturn_char(%[[res_addr:[^,]*]], %[[res_len:[^,]*]], {{.*}})
! CHECK: %[[res:.*]]:2 = hlfir.declare %[[res_addr]] typeparams %[[res_len]]
! CHECK: %[[val_18:.*]] = fir.embox %[[res]]#1 typeparams %[[res_len]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK: %[[val_19:.*]] = fir.box_elesize %[[val_18]] : (!fir.box<!fir.char<1,?>>) -> i32
! CHECK: %[[val_20:.*]] = arith.constant 8 : i32
! CHECK: %[[val_21:.*]] = arith.muli %[[val_19]], %[[val_20]] : i32
! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[val_21]])
end subroutine

function return_char(l)
  integer :: l
  character(l) :: return_char
end function

  call test_storage_size(42)
  print *, 42*8
end
