! Test lowering of references to pointers
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Assigning/reading to scalar pointer target.
! CHECK-LABEL: func @_QPscal_ptr(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>{{.*}})
subroutine scal_ptr(p)
  real, pointer :: p
  real :: x
  ! CHECK: %[[pdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFscal_ptrEp"}
  ! CHECK: %[[boxload:.*]] = fir.load %[[pdecl]]#0
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[boxload]]
  ! CHECK: hlfir.assign %{{.*}} to %[[addr]]
  p = 3.

  ! CHECK: %[[boxload2:.*]] = fir.load %[[pdecl]]#0
  ! CHECK: %[[addr2:.*]] = fir.box_addr %[[boxload2]]
  ! CHECK: %[[val:.*]] = fir.load %[[addr2]]
  ! CHECK: hlfir.assign %[[val]] to %{{.*}}
  x = p
end subroutine

! Assigning/reading scalar character pointer target.
! CHECK-LABEL: func @_QPchar_ptr(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,12>>>>{{.*}})
subroutine char_ptr(p)
  character(12), pointer :: p
  character(12) :: x

  ! CHECK: %[[pdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFchar_ptrEp"}
  ! CHECK: %[[straddr:.*]] = fir.address_of(@_QQclX68656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
  ! CHECK: %[[str:.*]]:2 = hlfir.declare %[[straddr]] typeparams %{{.*}} {{.*}} : (!fir.ref<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>)
  ! CHECK: %[[boxload:.*]] = fir.load %[[pdecl]]#0
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[boxload]]
  ! CHECK: hlfir.assign %[[str]]#0 to %[[addr]] : !fir.ref<!fir.char<1,12>>, !fir.ptr<!fir.char<1,12>>
  p = "hello world!"

  ! CHECK: %[[boxload2:.*]] = fir.load %[[pdecl]]#0
  ! CHECK: %[[addr2:.*]] = fir.box_addr %[[boxload2]]
  ! CHECK: hlfir.assign %[[addr2]] to %{{.*}} : !fir.ptr<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>
  x = p
end subroutine

! Reading from pointer in array expression
! CHECK-LABEL: func @_QParr_ptr_read(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}})
subroutine arr_ptr_read(p)
  real, pointer :: p(:)
  real :: x(100)
  ! CHECK: %[[pdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFarr_ptr_readEp"}
  ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFarr_ptr_readEx"}
  ! CHECK: %[[boxload:.*]] = fir.load %[[pdecl]]#0
  ! CHECK: hlfir.assign %[[boxload]] to %[[xdecl]]#0 : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.array<100xf32>>
  x = p
end subroutine

! Reading from contiguous pointer in array expression
! CHECK-LABEL: func @_QParr_contig_ptr_read(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {{{.*}}, fir.contiguous})
subroutine arr_contig_ptr_read(p)
  real, pointer, contiguous :: p(:)
  real :: x(100)
  ! CHECK: %[[pdecl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}} {fortran_attrs = #fir.var_attrs<contiguous, pointer>, uniq_name = "_QFarr_contig_ptr_readEp"}
  ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFarr_contig_ptr_readEx"}
  ! CHECK: %[[boxload:.*]] = fir.load %[[pdecl]]#0
  ! CHECK: hlfir.assign %[[boxload]] to %[[xdecl]]#0 : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.array<100xf32>>
  x = p
end subroutine

! Assigning to pointer target in array expression

  ! CHECK-LABEL: func @_QParr_ptr_target_write(
  ! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}) {
  ! CHECK:         %[[VAL_P:.*]]:2 = hlfir.declare %[[VAL_0]]{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFarr_ptr_target_writeEp"}
  ! CHECK:         %[[VAL_X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFarr_ptr_target_writeEx"}
  ! CHECK:         %[[VAL_PLOAD:.*]] = fir.load %[[VAL_P]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_DESIG:.*]] = hlfir.designate %[[VAL_PLOAD]] (%c2{{.*}}:%c601{{.*}}:%c6{{.*}})  shape %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
  ! CHECK:         hlfir.assign %[[VAL_X]]#0 to %[[VAL_DESIG]] : !fir.ref<!fir.array<100xf32>>, !fir.box<!fir.array<100xf32>>
  ! CHECK:         return
  ! CHECK:       }

subroutine arr_ptr_target_write(p)
  real, pointer :: p(:)
  real :: x(100)
  p(2:601:6) = x
end subroutine

! Assigning to contiguous pointer target in array expression

  ! CHECK-LABEL: func @_QParr_contig_ptr_target_write(
  ! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {{{.*}}, fir.contiguous}) {
  ! CHECK:         %[[VAL_P:.*]]:2 = hlfir.declare %[[VAL_0]]{{.*}} {fortran_attrs = #fir.var_attrs<contiguous, pointer>, uniq_name = "_QFarr_contig_ptr_target_writeEp"}
  ! CHECK:         %[[VAL_X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFarr_contig_ptr_target_writeEx"}
  ! CHECK:         %[[VAL_PLOAD:.*]] = fir.load %[[VAL_P]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_DESIG:.*]] = hlfir.designate %[[VAL_PLOAD]] (%c2{{.*}}:%c601{{.*}}:%c6{{.*}})  shape %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
  ! CHECK:         hlfir.assign %[[VAL_X]]#0 to %[[VAL_DESIG]] : !fir.ref<!fir.array<100xf32>>, !fir.box<!fir.array<100xf32>>
  ! CHECK:         return
  ! CHECK:       }

subroutine arr_contig_ptr_target_write(p)
  real, pointer, contiguous :: p(:)
  real :: x(100)
  p(2:601:6) = x
end subroutine

! CHECK-LABEL: func @_QPpointer_result_as_value
subroutine pointer_result_as_value()
  ! Test that function pointer results used as values are correctly loaded.
  interface
    function returns_int_pointer()
      integer, pointer :: returns_int_pointer
    end function
  end interface
! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = ".result"}
! CHECK:  %[[VAL_6:.*]] = fir.call @_QPreturns_int_pointer() {{.*}}: () -> !fir.box<!fir.ptr<i32>>
! CHECK:  fir.save_result %[[VAL_6]] to %[[VAL_0]] : !fir.box<!fir.ptr<i32>>, !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_RES:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_RES]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  fir.load %[[VAL_8]] : !fir.ptr<i32>
  print *, returns_int_pointer()
end subroutine
