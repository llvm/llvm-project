! Test evaluate::SetLength lowering (used to set a different length on a
! character storage around calls where the dummy and actual length differ).
! RUN: bbc -emit-fir -o - %s | FileCheck %s


subroutine takes_length_4(c)
  character c(3)*4
  !do i = 1,3
  print *, c(i)
  !enddo
end

! CHECK-LABEL: func @_QPfoo(
subroutine foo(c)
  character c(4)*3
  ! evaluate::Expr is: CALL s(%SET_LENGTH(c(1_8),4_8)) after semantics.
  call takes_length_4(c(1))
! CHECK:         %[[VAL_2:.*]] = arith.constant 4 : i64
! CHECK:         %[[VAL_6:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, i64) -> !fir.ref<!fir.char<1,3>>
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = fir.emboxchar %[[VAL_7]], %[[VAL_8]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPtakes_length_4(%[[VAL_9]]) : (!fir.boxchar<1>) -> ()
end subroutine

  character(3) :: c(4) = ["abc", "def", "ghi", "klm"]
  call foo(c)
end
