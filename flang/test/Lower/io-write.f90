! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! Test that IO item calls stackrestore in the right place 

! CHECK-LABEL: func.func @_QQmain() {
  character(3) string
  write(string,getstring(6))
! CHECK:  %[[Val_0:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:  %[[Const_3:.*]] = arith.constant 3 : index
! CHECK:  %[[Val_1:.*]] = fir.alloca !fir.char<1,3> {bindc_name = "string", uniq_name = "_QFEstring"}
! CHECK:  %[[Val_2:.*]] = fir.convert %[[Val_1]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
! CHECK:  %[[Val_3:.*]] = fir.convert %[[Const_3]] : (index) -> i64
! CHECK:  %[[Const_6:.*]] = arith.constant 6 : i32
! CHECK:  fir.store %[[Const_6]] to %[[Val_0]] : !fir.ref<i32>
! CHECK:  %[[Val_4:.*]] = fir.load %[[Val_0]] : !fir.ref<i32>
! CHECK:  %[[Val_5:.*]] = fir.convert %[[Val_4]] : (i32) -> i64
! CHECK:  %[[Val_6:.*]] = fir.convert %[[Val_5]] : (i64) -> index
! CHECK:  %[[Const_0:.*]] = arith.constant 0 : index
! CHECK:  %[[Val_7:.*]] = arith.cmpi sgt, %[[Val_6]], %[[Const_0]] : index
! CHECK:  %[[Val_8:.*]] = arith.select %[[Val_7]], %[[Val_6]], %[[Const_0]] : index
! CHECK:  %[[Val_9:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK:  %[[Val_10:.*]] = fir.alloca !fir.char<1,?>(%[[Val_8]] : index) {bindc_name = ".result"}
! CHECK:  %[[Val_11:.*]] = fir.call @_QFPgetstring(%[[Val_10]], %[[Val_8]], %[[Val_0]]) {{.*}}: (!fir.ref<!fir.char<1,?>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
! CHECK:  %[[Val_12:.*]] = fir.convert %[[Val_10]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:  %[[Val_13:.*]] = fir.convert %[[Val_8]] : (index) -> i64
! CHECK:  %[[Val_14:.*]] = fir.zero_bits !fir.box<none>
! CHECK:  %[[Const_0_i64:.*]] = arith.constant 0 : i64
! CHECK:  %[[Val_15:.*]] = fir.convert %[[Const_0_i64]] : (i64) -> !fir.ref<!fir.llvm_ptr<i8>>
! CHECK:  %[[Const_0_i64_0:.*]] = arith.constant 0 : i64
! CHECK:  %[[Val_16:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:  %[[Val_17:.*]] = fir.convert %[[Val_16]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:  %[[Val_18:.*]] = fir.call @_FortranAioBeginInternalFormattedOutput(%[[Val_2]], %[[Val_3]], %[[Val_12]], %[[Val_13]],
! %[[Val_14]], %[[Val_15]], %[[Const_0_i64_0]], %17, %{{.*}}) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, !fir.box<none>, !fir.ref<!fir.llvm_ptr<i8>>, i64, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:  %[[Val_19:.*]] = fir.call @_FortranAioEndIoStatement(%18) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:  llvm.intr.stackrestore %[[Val_9]] : !llvm.ptr
  if (string/="hi") stop 'FAIL'
contains
  function getstring(n) result(r)
    character(n) r
    r = '("hi")'
  end function
end
