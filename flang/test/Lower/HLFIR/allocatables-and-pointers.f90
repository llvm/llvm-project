! Test lowering of whole allocatable and pointers to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine passing_allocatable(x)
  interface
    subroutine takes_allocatable(y)
      real, allocatable :: y(:)
    end subroutine
    subroutine takes_array(y)
      real :: y(*)
    end subroutine
  end interface
  real, allocatable :: x(:)
  call takes_allocatable(x)
  call takes_array(x)
end subroutine
! CHECK-LABEL: func.func @_QPpassing_allocatable(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name =  {{.*}}Ex"}
! CHECK:  fir.call @_QPtakes_allocatable(%[[VAL_1]]#0) {{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_array(%[[VAL_4]]) {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> ()

subroutine passing_pointer(x)
  interface
    subroutine takes_pointer(y)
      real, pointer :: y(:)
    end subroutine
  end interface
  real, pointer :: x(:)
  call takes_pointer(x)
  call takes_pointer(NULL())
end subroutine
! CHECK-LABEL: func.func @_QPpassing_pointer(
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name =  {{.*}}Ex"}
! CHECK:  fir.call @_QPtakes_pointer(%[[VAL_2]]#0) {{.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> ()
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]] = fir.embox %[[VAL_3]](%[[VAL_5]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  fir.call @_QPtakes_pointer(%[[VAL_1]]) {{.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> ()

subroutine passing_contiguous_pointer(x)
  interface
    subroutine takes_array(y)
      real :: y(*)
    end subroutine
  end interface
  real, pointer, contiguous :: x(:)
  call takes_array(x)
end subroutine
! CHECK-LABEL: func.func @_QPpassing_contiguous_pointer(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<contiguous, pointer>, uniq_name =  {{.*}}Ex"}
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_array(%[[VAL_4]]) {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> ()

subroutine character_allocatable_cst_len(x)
  character(10), allocatable :: x
  call takes_char(x)
  call takes_char(x//"hello")
end subroutine
! CHECK-LABEL: func.func @_QPcharacter_allocatable_cst_len(
! CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] typeparams %[[VAL_1:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name =  {{.*}}Ex"}
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:  %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:  %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,10>>
! CHECK:  %[[VAL_7:.*]] = fir.emboxchar %[[VAL_6]], %[[VAL_5]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
! CHECK:  fir.call @_QPtakes_char(%[[VAL_7]]) {{.*}} : (!fir.boxchar<1>) -> ()
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:  %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_10:[a-z0-9]*]] typeparams %[[VAL_11:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<parameter>
! CHECK:  %[[VAL_13:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_11]] : index
! CHECK:  %[[VAL_15:.*]] = hlfir.concat %[[VAL_9]], %[[VAL_12]]#0 len %[[VAL_14]] : (!fir.heap<!fir.char<1,10>>, !fir.ref<!fir.char<1,5>>, index) -> !hlfir.expr<!fir.char<1,15>>

subroutine character_allocatable_dyn_len(x, l)
  integer(8) :: l
  character(l), allocatable :: x
  call takes_char(x)
  call takes_char(x//"hello")
end subroutine
! CHECK-LABEL: func.func @_QPcharacter_allocatable_dyn_len(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]] {uniq_name =  {{.*}}El"}
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] typeparams %[[VAL_6:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name =  {{.*}}Ex"}
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:  %[[VAL_10:.*]] = fir.emboxchar %[[VAL_9]], %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
! CHECK:  fir.call @_QPtakes_char(%[[VAL_10]]) {{.*}} : (!fir.boxchar<1>) -> ()
! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:  %[[VAL_13:.*]] = fir.emboxchar %[[VAL_12]], %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
! CHECK:  %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_14:[a-z0-9]*]] typeparams %[[VAL_15:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<parameter>
! CHECK:  %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:  %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_15]] : index
! CHECK:  %[[VAL_19:.*]] = hlfir.concat %[[VAL_13]], %[[VAL_16]]#0 len %[[VAL_18]] : (!fir.boxchar<1>, !fir.ref<!fir.char<1,5>>, index) -> !hlfir.expr<!fir.char<1,?>>

subroutine print_allocatable(x)
  real, allocatable :: x(:)
  print *, x
end subroutine
! CHECK-LABEL: func.func @_QPprint_allocatable(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name =  {{.*}}Ex"}
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_1]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.box<none>
! CHECK:  %[[VAL_9:.*]] = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[VAL_8]])

subroutine print_pointer(x)
  real, pointer :: x(:)
  print *, x
end subroutine
! CHECK-LABEL: func.func @_QPprint_pointer(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name =  {{.*}}Ex"}
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_1]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
! CHECK:  %[[VAL_9:.*]] = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[VAL_8]])

subroutine elemental_expr(x)
  integer, pointer :: x(:, :)
  call takes_array_2(x+42)
end subroutine
! CHECK-LABEL: func.func @_QPelemental_expr(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name =  {{.*}}Ex"}
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 42 : i32
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?x?xi32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?x?xi32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_5]]#1, %[[VAL_7]]#1 : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_9:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
! CHECK:  ^bb0(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
! CHECK:    %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_12]] : (!fir.box<!fir.ptr<!fir.array<?x?xi32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_14]] : (!fir.box<!fir.ptr<!fir.array<?x?xi32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_17:.*]] = arith.subi %[[VAL_13]]#0, %[[VAL_16]] : index
! CHECK:    %[[VAL_18:.*]] = arith.addi %[[VAL_10]], %[[VAL_17]] : index
! CHECK:    %[[VAL_19:.*]] = arith.subi %[[VAL_15]]#0, %[[VAL_16]] : index
! CHECK:    %[[VAL_20:.*]] = arith.addi %[[VAL_11]], %[[VAL_19]] : index
! CHECK:    %[[VAL_21:.*]] = hlfir.designate %[[VAL_2]] (%[[VAL_18]], %[[VAL_20]])  : (!fir.box<!fir.ptr<!fir.array<?x?xi32>>>, index, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
! CHECK:    %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : i32
! CHECK:    hlfir.yield_element %[[VAL_23]] : i32
! CHECK:  }
