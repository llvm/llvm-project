! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test that IO item list

! FIXME: embox does not like getting a length when it gets
! a !fir.ref<!fir.char<kind>> buffer. Either the verifier
! should be relaxed, or we should finish up ensuring character
! type for such buffer are !fir.ref<fir.array<?x!fir.char<kind>>>
!
!subroutine pass_assumed_len_char(c)
!  character(*) :: c
!  write(1, rec=1) c
!end

! CHECK-LABEL: func @_QPpass_assumed_len_char_array 
subroutine pass_assumed_len_char_array(carray)
  character(*) :: carray(2, 3)
  ! CHECK-DAG: %[[unboxed:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, index)
  ! CHECK-DAG: %[[buffer:.*]] = fir.convert %[[unboxed]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.array<?x2x3x!fir.char<1>>>
  ! CHECK-DAG: %[[c2:.*]] = constant 2 : index
  ! CHECK-DAG: %[[c3:.*]] = constant 3 : index
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %[[c2]], %[[c3]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[box:.*]] = fir.embox %[[buffer]](%[[shape]]) typeparams %[[unboxed]]#1 : (!fir.ref<!fir.array<?x2x3x!fir.char<1>>>, !fir.shape<2>, index) -> !fir.box<!fir.array<?x2x3x!fir.char<1>>>
  ! CHECK: %[[descriptor:.*]] = fir.convert %[[box]] : (!fir.box<!fir.array<?x2x3x!fir.char<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[descriptor]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  print *, carray
end
