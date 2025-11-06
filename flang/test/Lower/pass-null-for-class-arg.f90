! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefix=FIR
! RUN: bbc -emit-fir -hlfir %s -o - | FileCheck %s --check-prefix=HLFIR

subroutine test
  interface
     subroutine s1p(n)
       type t
          integer :: n
       end type t
       class(t), pointer :: n
     end subroutine s1p
  end interface
  call s1p(null())
end subroutine test
! FIR-LABEL:   func.func @_QPtest() {
! FIR:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>>
! FIR:           %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>
! FIR:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>) -> !fir.class<!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>>
! FIR:           fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>>>
! FIR:           fir.call @_QPs1p(%[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>>>) -> ()
! FIR:           return
! FIR:         }

! HLFIR-LABEL:   func.func @_QPtest() {
! HLFIR:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>>
! HLFIR:           %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>
! HLFIR:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>) -> !fir.class<!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>>
! HLFIR:           fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>>>
! HLFIR:           fir.call @_QPs1p(%[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QFtestFs1pTt{n:i32}>>>>) -> ()
! HLFIR:           return
! HLFIR:         }
