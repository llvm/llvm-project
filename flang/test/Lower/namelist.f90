! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
program p
  ! CHECK-DAG: [[ccc:%[0-9]+]] = fir.address_of(@_QEccc) : !fir.ref<!fir.array<4x!fir.char<1,3>>>
  ! CHECK-DAG: [[jjj:%[0-9]+]] = fir.alloca i32 {bindc_name = "jjj", uniq_name = "_QEjjj"}
  character*3 ccc(4)
  namelist /nnn/ jjj, ccc
  jjj = 17
  ccc = ["aa ", "bb ", "cc ", "dd "]
  ! CHECK: [[cookie:%[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: fir.alloca !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK: fir.undefined
  ! CHECK: fir.address_of
  ! CHECK: fir.insert_value
  ! CHECK: fir.embox [[jjj]]
  ! CHECK: fir.insert_value
  ! CHECK: fir.address_of
  ! CHECK: fir.insert_value
  ! CHECK: fir.embox [[ccc]]
  ! CHECK: fir.insert_value
  ! CHECK: fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>>
  ! CHECK: fir.address_of
  ! CHECK-COUNT-3: fir.insert_value
  ! CHECK: fir.call @_FortranAioOutputNamelist([[cookie]]
  ! CHECK: fir.call @_FortranAioEndIoStatement([[cookie]]
  write(*, nnn)
  jjj = 27
  ! CHECK: fir.coordinate_of
  ccc(4) = "zz "
  ! CHECK: [[cookie:%[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: fir.alloca !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK: fir.undefined
  ! CHECK: fir.address_of
  ! CHECK: fir.insert_value
  ! CHECK: fir.embox [[jjj]]
  ! CHECK: fir.insert_value
  ! CHECK: fir.address_of
  ! CHECK: fir.insert_value
  ! CHECK: fir.embox [[ccc]]
  ! CHECK: fir.insert_value
  ! CHECK: fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>>
  ! CHECK: fir.address_of
  ! CHECK-COUNT-3: fir.insert_value
  ! CHECK: fir.call @_FortranAioOutputNamelist([[cookie]]
  ! CHECK: fir.call @_FortranAioEndIoStatement([[cookie]]
  write(*, nnn)
end
  ! CHECK-DAG: fir.global linkonce @_QQcl.6A6A6A00 constant : !fir.char<1,4>
  ! CHECK-DAG: fir.global linkonce @_QQcl.63636300 constant : !fir.char<1,4>
  ! CHECK-DAG: fir.global linkonce @_QQcl.6E6E6E00 constant : !fir.char<1,4>
