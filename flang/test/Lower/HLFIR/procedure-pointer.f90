! test level 1 procedure pointer for
! 1. declaration and initialization
! 2. pointer assignment and invocation
! 3. procedure pointer argument passing.
! 3. procedure pointer function result.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module m
  interface
    real function real_func(x)
      real :: x
    end function
    character(:) function char_func(x)
      pointer :: char_func 
      integer :: x
    end function
    subroutine sub(x)
      real :: x
    end subroutine
    subroutine foo2(q)
      import
      procedure(char_func), pointer :: q
    end
  end interface

end module m

!!! Testing declaration and initialization
subroutine  sub1()
use m
  procedure(real_func), pointer :: p1
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<(!fir.ref<f32>) -> f32> {bindc_name = "p1", uniq_name = "_QFsub1Ep1"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub1Ep1"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)

  procedure(real_func), pointer :: p2 => null()
! CHECK: %[[VAL_0:.*]] = fir.address_of(@_QFsub1Ep2) : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub1Ep2"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)

  procedure(real_func), pointer :: p3 => real_func
! CHECK: %[[VAL_0:.*]] = fir.address_of(@_QFsub1Ep3) : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub1Ep3"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)

  procedure(), pointer :: p4
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<() -> ()> {bindc_name = "p4", uniq_name = "_QFsub1Ep4"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits () -> ()
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub1Ep4"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)

  procedure(real), pointer :: p5
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<() -> f32> {bindc_name = "p5", uniq_name = "_QFsub1Ep5"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits () -> f32
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : (() -> f32) -> !fir.boxproc<() -> f32>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<() -> f32>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub1Ep5"} : (!fir.ref<!fir.boxproc<() -> f32>>) -> (!fir.ref<!fir.boxproc<() -> f32>>, !fir.ref<!fir.boxproc<() -> f32>>)

  procedure(char_func), pointer :: p6
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>> {bindc_name = "p6", uniq_name = "_QFsub1Ep6"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub1Ep6"} : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>, !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>)

  procedure(char_func), pointer :: p7 => char_func
! CHECK: %[[VAL_0:.*]] = fir.address_of(@_QFsub1Ep7) : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub1Ep7"} : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>, !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>)
end subroutine sub1


!!! Testing pointer assignment and invocation
subroutine  sub2()
use m
  procedure(real_func), pointer :: p1

  p1 => null()
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<(!fir.ref<f32>) -> f32> {bindc_name = "p1", uniq_name = "_QFsub2Ep1"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub2Ep1"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)
! CHECK: %[[VAL_4:.*]] = fir.zero_bits () -> ()
! CHECK: %[[VAL_5:.*]] = fir.emboxproc %[[VAL_4]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_6]] to %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
end subroutine

subroutine  sub3()
use m
  procedure(real_func), pointer :: p1
  real :: res, r

  p1 => real_func
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<(!fir.ref<f32>) -> f32> {bindc_name = "p1", uniq_name = "_QFsub3Ep1"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub3Ep1"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)
! CHECK: %[[VAL_4:.*]] = fir.address_of(@_QPreal_func) : (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_5:.*]] = fir.emboxproc %[[VAL_4]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_6]] to %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>

  res = p1(r)
! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.boxproc<(!fir.ref<f32>) -> f32>) -> ((!fir.ref<f32>) -> f32)
! CHECK: %[[VAL_9:.*]] = fir.call %[[VAL_8]](%5#1) fastmath<contract> : (!fir.ref<f32>) -> f32

  nullify(p1)
! CHECK: %[[VAL_10:.*]] = fir.zero_bits () -> ()
! CHECK: %[[VAL_11:.*]] = fir.emboxproc %[[VAL_10]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_12]] to %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
end subroutine

subroutine  sub4()
use m
  procedure(char_func), pointer :: p2
  character(:), pointer :: res
  integer :: i

  p2 => char_func
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>> {bindc_name = "p2", uniq_name = "_QFsub4Ep2"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub4Ep2"} : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>, !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>)
! CHECK: %[[VAL_4:.*]] = fir.address_of(@_QPchar_func) : (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK: %[[VAL_12:.*]] = arith.constant -1 : index
! CHECK: %[[VAL_5:.*]] = fir.emboxproc %[[VAL_4]] : ((!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
! CHECK: %[[VAL_7:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_5]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_6]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[VAL_10:.*]] = fir.extract_value %[[VAL_9]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: fir.store %[[VAL_11]] to %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>

  res = p2(i)
! CHECK: %[[VAL_12:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK: %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>) -> ((!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>)
! CHECK: %[[VAL_14:.*]] = fir.call %[[VAL_13]](%2#1) fastmath<contract> : (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
end subroutine

subroutine  sub5()
use m
  procedure(real), pointer :: p3

  p3 => real_func 
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<() -> f32> {bindc_name = "p3", uniq_name = "_QFsub5Ep3"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits () -> f32
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : (() -> f32) -> !fir.boxproc<() -> f32>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<() -> f32>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub5Ep3"} : (!fir.ref<!fir.boxproc<() -> f32>>) -> (!fir.ref<!fir.boxproc<() -> f32>>, !fir.ref<!fir.boxproc<() -> f32>>)
! CHECK: %[[VAL_4:.*]] = fir.address_of(@_QPreal_func) : (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_5:.*]] = fir.emboxproc %[[VAL_4]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<() -> f32>
! CHECK: fir.store %[[VAL_6]] to %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<() -> f32>>
end subroutine

subroutine  sub6()
use m
  procedure(), pointer :: p4
  real :: r

  p4 => sub 
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<() -> ()> {bindc_name = "p4", uniq_name = "_QFsub6Ep4"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits () -> ()
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub6Ep4"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK: %[[VAL_4:.*]] = fir.address_of(@_QPsub) : (!fir.ref<f32>) -> ()
! CHECK: %[[VAL_5:.*]] = fir.emboxproc %[[VAL_4]] : ((!fir.ref<f32>) -> ()) -> !fir.boxproc<() -> ()>
! CHECK: fir.store %[[VAL_5]] to %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<() -> ()>>

  call p4(r)
! CHECK: %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK: %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<f32>) -> ())
! CHECK: fir.call %[[VAL_7]](%5#1) fastmath<contract> : (!fir.ref<f32>) -> ()
end subroutine


!!! Testing pointer assignment and invocation
subroutine sub7(p1, p2)
use m
  procedure(real_func), pointer :: p1
! CHECK: %[[VAL_0:.*]]:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub7Ep1"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)

  procedure(char_func), pointer :: p2
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub7Ep2"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)

  call foo1(p1)
! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK: fir.call @_QPfoo1(%[[VAL_2]]) fastmath<contract> : (!fir.boxproc<() -> ()>) -> ()

  call foo2(p2)
! CHECK: fir.call @_QPfoo2(%[[VAL_1]]#0) fastmath<contract> : (!fir.ref<!fir.boxproc<() -> ()>>) -> ()
end 

subroutine sub8()
use m
  procedure(real_func), pointer, save :: pp1
! CHECK: %[[VAL_0:.*]] = fir.address_of(@_QFsub8Epp1) : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub8Epp1"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)

  procedure(char_func), pointer, save :: pp2
! CHECK: %[[VAL_2:.*]] = fir.address_of(@_QFsub8Epp2) : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub8Epp2"} : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>, !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>)

  call foo1(pp1)
! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.boxproc<(!fir.ref<f32>) -> f32>) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPfoo1(%[[VAL_5]]) fastmath<contract> : (!fir.boxproc<() -> ()>) -> ()

  call foo2(pp2)
! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>) -> !fir.ref<!fir.boxproc<() -> ()>>
! CHECK: fir.call @_QPfoo2(%[[VAL_6]]) fastmath<contract> : (!fir.ref<!fir.boxproc<() -> ()>>) -> ()
end

subroutine sub9()
use m
  procedure(real_func), pointer :: p1
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<(!fir.ref<f32>) -> f32> {bindc_name = "p1", uniq_name = "_QFsub9Ep1"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub9Ep1"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)

  procedure(char_func), pointer :: p2
! CHECK: %[[VAL_4:.*]] = fir.alloca !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>> {bindc_name = "p2", uniq_name = "_QFsub9Ep2"}
! CHECK: %[[VAL_5:.*]] = fir.zero_bits (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK: %[[VAL_6:.*]] = fir.emboxproc %[[VAL_5]] : ((!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: fir.store %[[VAL_6]] to %[[VAL_4]] : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub9Ep2"} : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>, !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>)

  call foo1(p1)
! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.boxproc<(!fir.ref<f32>) -> f32>) -> !fir.boxproc<() -> ()>
! CHECK: fir.call @_QPfoo1(%[[VAL_9]]) fastmath<contract> : (!fir.boxproc<() -> ()>) -> ()

  call foo2(p2)
! CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>) -> !fir.ref<!fir.boxproc<() -> ()>>
! CHECK: fir.call @_QPfoo2(%[[VAL_10]]) fastmath<contract> : (!fir.ref<!fir.boxproc<() -> ()>>) -> ()
end

subroutine sub10()
use m

  procedure(real_func), pointer :: p1
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<(!fir.ref<f32>) -> f32> {bindc_name = "p1", uniq_name = "_QFsub10Ep1"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub10Ep1"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)

  p1 => reffunc(5)
! CHECK: %c5_i32 = arith.constant 5 : i32
! CHECK: %[[VAL_4:.*]]:3 = hlfir.associate %c5_i32 {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK: %[[VAL_5:.*]] = fir.call @_QFsub10Preffunc(%[[VAL_4]]#1) fastmath<contract> : (!fir.ref<i32>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_5]] to %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>

contains
  function reffunc(arg) result(pp)
    integer :: arg
    procedure(real_func), pointer :: pp
! CHECK: %[[VAL_0:.*]]:2 = hlfir.declare %arg0 {uniq_name = "_QFsub10FreffuncEarg"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[VAL_1:.*]] = fir.alloca !fir.boxproc<(!fir.ref<f32>) -> f32> {bindc_name = "pp", uniq_name = "_QFsub10FreffuncEpp"}
! CHECK: %[[VAL_2:.*]] = fir.zero_bits (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_3:.*]] = fir.emboxproc %[[VAL_2]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub10FreffuncEpp"} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)

    pp => real_func
! CHECK: %[[VAL_5:.*]] = fir.address_of(@_QPreal_func) : (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_6:.*]] = fir.emboxproc %[[VAL_5]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.store %[[VAL_7]] to %[[VAL_4]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_4]]#1 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
! CHECK: return %[[VAL_8]] : !fir.boxproc<(!fir.ref<f32>) -> f32>
  end
end

subroutine sub11()
use m
  interface
    function reffunc(arg) result(pp)
      import
      integer :: arg
      procedure(char_func), pointer :: pp
    end
  end interface

  procedure(char_func), pointer :: p1
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>> {bindc_name = "p1", uniq_name = "_QFsub11Ep1"}
! CHECK: %[[VAL_1:.*]] = fir.zero_bits (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK: %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsub11Ep1"} : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>, !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>)

  p1 => reffunc(5)
! CHECK: %c5_i32 = arith.constant 5 : i32
! CHECK: %[[VAL_4:.*]]:3 = hlfir.associate %c5_i32 {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK: %[[VAL_5:.*]] = fir.call @_QPreffunc(%4#1) fastmath<contract> : (!fir.ref<i32>) -> !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: fir.store %[[VAL_5]] to %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK: return
end

! CHECK-LABEL: fir.global internal @_QFsub1Ep2 : !fir.boxproc<(!fir.ref<f32>) -> f32> {
! CHECK: %[[VAL_0:.*]] = fir.zero_bits (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.has_value %[[VAL_1]] : !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: }

! CHECK-LABEL: fir.global internal @_QFsub1Ep3 : !fir.boxproc<(!fir.ref<f32>) -> f32> {
! CHECK: %[[VAL_0:.*]] = fir.address_of(@_QPreal_func) : (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.has_value %[[VAL_2]] : !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: }

! CHECK-LABEL: fir.global internal @_QFsub1Ep7 : !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>> {
! CHECK: %[[VAL_0:.*]] = fir.address_of(@_QPchar_func) : (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK: %[[VAL_11:.*]] = arith.constant -1 : index
! CHECK: %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_2:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
! CHECK: %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_1]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_2]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[VAL_6:.*]] = fir.extract_value %[[VAL_5]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_7:.*]] = fir.extract_value %[[VAL_5]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: fir.has_value %[[VAL_8]] : !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: }

! CHECK-LABEL:  fir.global internal @_QFsub8Epp1 : !fir.boxproc<(!fir.ref<f32>) -> f32> {
! CHECK: %[[VAL_0:.*]] = fir.zero_bits (!fir.ref<f32>) -> f32
! CHECK: %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: fir.has_value %[[VAL_1]] : !fir.boxproc<(!fir.ref<f32>) -> f32>
! CHECK: }

! CHECK-LABEL:  fir.global internal @_QFsub8Epp2 : !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>> {
! CHECK: %[[VAL_0:.*]] = fir.zero_bits (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK: %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: fir.has_value %[[VAL_1]] : !fir.boxproc<(!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK: }
