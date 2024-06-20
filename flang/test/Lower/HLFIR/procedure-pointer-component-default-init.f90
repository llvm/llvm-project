! Test procedure pointer component default initialization when the size
! of the derived type is 32 bytes and larger. 
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

  interface
    subroutine sub()
    end
  end interface
  type dt
    real :: r1 = 5.0
    procedure(real), pointer, nopass :: pp1 => null()
    real, pointer :: rp1 => null()
    procedure(), pointer, nopass :: pp2 => sub
  end type
  type(dt) :: dd1
  end

! CHECK-LABEL: func.func @_QQmain() {
! CHECK:    %[[VAL_14:.*]] = fir.address_of(@_QFEdd1) : !fir.ref<!fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>>
! CHECK:    %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] {uniq_name = "_QFEdd1"} : (!fir.ref<!fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>>) -> (!fir.ref<!fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>>, !fir.ref<!fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>>)
! CHECK:  }

! CHECK-LABEL:  fir.global internal @_QFEdd1 : !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}> {
! CHECK:    %[[VAL_0:.*]] = fir.undefined !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:    %cst = arith.constant 5.000000e+00 : f32
! CHECK:    %[[VAL_1:.*]] = fir.field_index r1, !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:    %[[VAL_2:.*]] = fir.insert_value %[[VAL_0]], %cst, ["r1", !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>] : (!fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>, f32) -> !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:    %[[VAL_3:.*]] = fir.zero_bits () -> f32
! CHECK:    %[[VAL_4:.*]] = fir.emboxproc %[[VAL_3]] : (() -> f32) -> !fir.boxproc<() -> f32>
! CHECK:    %[[VAL_5:.*]] = fir.field_index pp1, !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:    %[[VAL_6:.*]] = fir.insert_value %[[VAL_2]], %[[VAL_4]], ["pp1", !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>] : (!fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>, !fir.boxproc<() -> f32>) -> !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:    %[[VAL_7:.*]] = fir.zero_bits !fir.ptr<f32>
! CHECK:    %[[VAL_8:.*]] = fir.embox %[[VAL_7]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK:    %[[VAL_9:.*]] = fir.field_index rp1, !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:    %[[VAL_10:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_8]], ["rp1", !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>] : (!fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>, !fir.box<!fir.ptr<f32>>) -> !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:    %[[VAL_11:.*]] = fir.address_of(@_QPsub) : () -> ()
! CHECK:    %[[VAL_12:.*]] = fir.emboxproc %[[VAL_11]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:    %[[VAL_13:.*]] = fir.field_index pp2, !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:    %[[VAL_14:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_12]], ["pp2", !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>] : (!fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>, !fir.boxproc<() -> ()>) -> !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:    fir.has_value %[[VAL_14]] : !fir.type<_QFTdt{r1:f32,pp1:!fir.boxproc<() -> f32>,rp1:!fir.box<!fir.ptr<f32>>,pp2:!fir.boxproc<() -> ()>}>
! CHECK:  }
