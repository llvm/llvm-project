! Test procedure pointers with the same name as generics.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module m_gen
  procedure(func), pointer :: foo
  interface foo
     procedure :: foo
  end interface
  interface
    real function func(x)
      real :: x
    end function
  end interface
end
!CHECK-LABEL:   fir.global @_QMm_genEfoo : !fir.boxproc<(!fir.ref<f32>) -> f32> {
!CHECK:           %[[VAL_0:.*]] = fir.zero_bits (!fir.ref<f32>) -> f32
!CHECK:           %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
!CHECK:           fir.has_value %[[VAL_1]] : !fir.boxproc<(!fir.ref<f32>) -> f32>

subroutine test1()
  use m_gen
  foo => func
end subroutine
!CHECK-LABEL:   func.func @_QPtest1() {
!CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QMm_genEfoo) : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
!CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}"_QMm_genEfoo"{{.*}} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)
!CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QPfunc) : (!fir.ref<f32>) -> f32
!CHECK:           %[[VAL_3:.*]] = fir.emboxproc %[[VAL_2]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
!CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
!CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>

subroutine test_local()
  use m_gen, only : func
  procedure(func), pointer :: foo
  interface foo
     procedure :: foo
  end interface
  foo => func
end subroutine
!CHECK-LABEL:   func.func @_QPtest_local() {
!CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.boxproc<(!fir.ref<f32>) -> f32> {bindc_name = "foo", uniq_name = "_QFtest_localEfoo"}
!CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}"_QFtest_localEfoo"{{.*}} : (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>) -> (!fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>, !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>)
!CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QPfunc) : (!fir.ref<f32>) -> f32
!CHECK:           %[[VAL_5:.*]] = fir.emboxproc %[[VAL_4]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
!CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<f32>) -> f32>
!CHECK:           fir.store %[[VAL_6]] to %[[VAL_3]]#0 : !fir.ref<!fir.boxproc<(!fir.ref<f32>) -> f32>>
