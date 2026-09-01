! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test that IO item calls stackrestore in the right place

! CHECK-LABEL: func.func @_QQmain() {
  character(3) string
  write(string,getstring(6))
! CHECK:  %[[Const_3:.*]] = arith.constant 3 : index
! CHECK:  %[[Val_1:.*]] = fir.alloca !fir.char<1,3> {bindc_name = "string"
! CHECK:  %[[string:.*]]:2 = hlfir.declare %[[Val_1]] typeparams %[[Const_3]]
! CHECK:  %[[Val_2:.*]] = fir.convert %[[string]]#0 : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
! CHECK:  %[[Val_3:.*]] = fir.convert %[[Const_3]] : (index) -> i64
! CHECK:  %[[n_assoc:.*]]:3 = hlfir.associate %{{.*}} {adapt.valuebyref}
! CHECK:  %[[Val_8:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
! CHECK:  %[[Val_9:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK:  %[[Val_10:.*]] = fir.alloca !fir.char<1,?>(%[[Val_8]] : index) {bindc_name = ".result"}
! CHECK:  fir.call @_QFPgetstring(%[[Val_10]], %[[Val_8]], %[[n_assoc]]#0)
! CHECK:  %[[Val_18:.*]] = fir.call @_FortranAioBeginInternalFormattedOutput(%[[Val_2]], %[[Val_3]],
! CHECK:  %[[Val_19:.*]] = fir.call @_FortranAioEndIoStatement(%[[Val_18]])
! CHECK:  llvm.intr.stackrestore %[[Val_9]] : !llvm.ptr
  if (string/="hi") stop 'FAIL'
contains
  function getstring(n) result(r)
    character(n) r
    r = '("hi")'
  end function
end
