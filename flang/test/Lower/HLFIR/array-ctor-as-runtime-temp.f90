! Test lowering of array constructors requiring runtime library help to HLFIR.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
module arrayctor
contains

subroutine test_loops()
  call takes_int([((i, i=1,ifoo()), j=1,ibar())])
end subroutine
! CHECK-LABEL:   func.func @_QMarrayctorPtest_loops() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<10xi64> {bindc_name = ".rt.arrayctor.vector"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = ".tmp.arrayctor"}
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_4]](%[[VAL_5]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_7:.*]] = arith.constant false
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_11:.*]] = arith.constant 7 : i32
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_14:.*]] = fir.call @_FortranAInitArrayConstructorVector(%[[VAL_8]], %[[VAL_12]], %[[VAL_7]], %[[VAL_13]], %[[VAL_11]]) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
! CHECK:           %[[VAL_17:.*]] = fir.call @_QMarrayctorPibar() fastmath<contract> : () -> i32
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           fir.do_loop %[[VAL_22:.*]] = %[[VAL_16]] to %[[VAL_19]] step %[[VAL_21]] {
! CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
! CHECK:             %[[VAL_25:.*]] = fir.call @_QMarrayctorPifoo() fastmath<contract> : () -> i32
! CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
! CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:             %[[VAL_28:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
! CHECK:             fir.do_loop %[[VAL_30:.*]] = %[[VAL_24]] to %[[VAL_27]] step %[[VAL_29]] {
! CHECK:               %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
! CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i64) -> i32
! CHECK:               fir.store %[[VAL_32]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:               %[[VAL_33:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:               %[[VAL_34:.*]] = fir.call @_FortranAPushArrayConstructorSimpleScalar(%[[VAL_8]], %[[VAL_33]]) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>) -> none
! CHECK:             }
! CHECK:           }
! CHECK:           %[[VAL_35:.*]] = arith.constant true
! CHECK:           %[[VAL_36:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_37:.*]] = hlfir.as_expr %[[VAL_36]] move %[[VAL_35]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, i1) -> !hlfir.expr<?xi32>
! CHECK:           %[[VAL_38:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_39:.*]]:3 = fir.box_dims %[[VAL_36]], %[[VAL_38]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_40:.*]] = fir.shape %[[VAL_39]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_41:.*]]:3 = hlfir.associate %[[VAL_37]](%[[VAL_40]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:           fir.call @_QMarrayctorPtakes_int(%[[VAL_41]]#0) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_41]]#1, %[[VAL_41]]#2 : !fir.ref<!fir.array<?xi32>>, i1
! CHECK:           hlfir.destroy %[[VAL_37]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

subroutine test_arrays(a)
  integer :: a(:, :)
  call takes_int([a, a])
end subroutine
! CHECK-LABEL: func.func @_QMarrayctorPtest_arrays(
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.array<10xi64> {bindc_name = ".rt.arrayctor.vector"}
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = ".tmp.arrayctor"}
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Ea"
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_5]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
! CHECK:  %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_8]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#1 : (index) -> i64
! CHECK:  %[[VAL_11:.*]] = arith.muli %[[VAL_7]], %[[VAL_10]] : i64
! CHECK:  %[[VAL_12:.*]] = arith.addi %[[VAL_4]], %[[VAL_11]] : i64
! CHECK:  %[[VAL_20:.*]] = arith.addi %[[VAL_12]], %{{.*}} : i64
! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:  %[[VAL_22:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_21]] {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:  %[[VAL_23:.*]] = fir.shape %[[VAL_21]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_22]](%[[VAL_23]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:  %[[VAL_25:.*]] = fir.embox %[[VAL_24]]#1(%[[VAL_23]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:  fir.store %[[VAL_25]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_26:.*]] = arith.constant false
! CHECK:  %[[VAL_27:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
! CHECK:  %[[VAL_31:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[VAL_33:.*]] = fir.call @_FortranAInitArrayConstructorVector(%[[VAL_27]], %[[VAL_31]], %[[VAL_26]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, !fir.ref<i8>, i32) -> none
! CHECK:  %[[VAL_34:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
! CHECK:  %[[VAL_35:.*]] = fir.call @_FortranAPushArrayConstructorValue(%[[VAL_27]], %[[VAL_34]]) {{.*}}: (!fir.llvm_ptr<i8>, !fir.box<none>) -> none
! CHECK:  %[[VAL_36:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
! CHECK:  %[[VAL_37:.*]] = fir.call @_FortranAPushArrayConstructorValue(%[[VAL_27]], %[[VAL_36]]) {{.*}}: (!fir.llvm_ptr<i8>, !fir.box<none>) -> none
! CHECK:  %[[VAL_38:.*]] = arith.constant true
! CHECK:  hlfir.as_expr %[[VAL_24]]#0 move %[[VAL_38]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>

subroutine test_arrays_unpredictable_size()
  call takes_int([rank1(), rank3(), rank1()])
! CHECK-LABEL: func.func @_QMarrayctorPtest_arrays_unpredictable_size() {
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.array<10xi64> {bindc_name = ".rt.arrayctor.vector"}
! CHECK:  %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = ".tmp.arrayctor"}
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_8:.*]] = fir.embox %[[VAL_6]](%[[VAL_7]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:  fir.store %[[VAL_8]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_9:.*]] = arith.constant false
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[VAL_16:.*]] = fir.call @_FortranAInitArrayConstructorVector(%[[VAL_10]], %[[VAL_14]], %[[VAL_9]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, !fir.ref<i8>, i32) -> none
! CHECK:  fir.call @_QMarrayctorPrank1() {{.*}}: () -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:  %[[VAL_21:.*]] = fir.call @_FortranAPushArrayConstructorValue(%[[VAL_10]], %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.box<none>) -> none
! CHECK:  fir.call @_QMarrayctorPrank3() {{.*}}: () -> !fir.box<!fir.heap<!fir.array<?x?x?xi32>>>
! CHECK:  %[[VAL_26:.*]] = fir.call @_FortranAPushArrayConstructorValue(%[[VAL_10]], %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.box<none>) -> none
! CHECK:  fir.call @_QMarrayctorPrank1() {{.*}}: () -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:  %[[VAL_31:.*]] = fir.call @_FortranAPushArrayConstructorValue(%[[VAL_10]], %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.box<none>) -> none
! CHECK:  %[[VAL_32:.*]] = arith.constant true
! CHECK:  %[[VAL_33:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:  hlfir.as_expr %[[VAL_33]] move %[[VAL_32]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, i1) -> !hlfir.expr<?xi32>
end subroutine


! End to to end test implementation
function rank1()
  integer, save :: counter = 2
  integer, allocatable :: rank1(:)
  allocate(rank1(counter))
  do i=1,counter
    rank1(i)=i
  end do
  counter = counter +1
end function
function rank3()
  integer, save :: counter = 1
  integer, allocatable :: rank3(:, :, :)
  allocate(rank3(counter, counter+1, counter+2))
  do k=1,counter+2
    do j=1,counter+1
      do i=1,counter
        rank3(i, j, k)=i+(j-1)*counter+(k-1)*counter*(counter+1)
      end do
    end do
  end do
  counter = counter+1
end function

function ifoo()
  integer, save :: counter = 0
  ifoo = counter
  counter = counter +1
end function

function ibar()
  ibar = 6
end function


subroutine takes_int(a)
  integer :: a(:)
  print *, "got   :", a
end subroutine
end module

  use arrayctor
  integer :: a(2,3) = reshape([1,2,3,4,5,6], shape=[2,3])
  print *, "expect: 1 1 2 1 2 3 1 2 3 4 1 2 3 4 5"
  call test_loops()
  print *, "expect: 1 2 3 4 5 6 1 2 3 4 5 6"
  call test_arrays(a)
  print *, "expect: 1 2 1 2 3 4 5 6 1 2 3"
  call test_arrays_unpredictable_size()
end
