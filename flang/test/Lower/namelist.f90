! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
program p
  ! CHECK-DAG: [[ccc:%[0-9]+]] = fir.alloca !fir.array<4x!fir.char<1,3>> {bindc_name = "ccc", uniq_name = "_QFEccc"}
  ! CHECK-DAG: [[jjj:%[0-9]+]] = fir.alloca i32 {bindc_name = "jjj", uniq_name = "_QFEjjj"}
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
  ! CHECK: fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK: fir.address_of
  ! CHECK-COUNT-4: fir.insert_value
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
  ! CHECK: fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK: fir.address_of
  ! CHECK-COUNT-4: fir.insert_value
  ! CHECK: fir.call @_FortranAioOutputNamelist([[cookie]]
  ! CHECK: fir.call @_FortranAioEndIoStatement([[cookie]]
  write(*, nnn)
end

! CHECK-LABEL: sss
subroutine sss
  integer xxx(11:13)
  namelist /rrr/ xxx
  ! CHECK: [[xxx:%[0-9]+]] = fir.alloca {{.*}} = "xxx"
  ! CHECK: [[cookie:%[0-9]+]] = fir.call @_FortranAioBeginExternalListInput
  ! CHECK: alloca
  ! CHECK: undefined
  ! CHECK: fir.address_of{{.*}}787878
  ! CHECK: fir.insert_value
  ! CHECK: fir.shape_shift %c11
  ! CHECK: fir.embox [[xxx]]
  ! CHECK: fir.insert_value
  ! CHECK: fir.alloca
  ! CHECK: fir.undefined
  ! CHECK: fir.address_of{{.*}}727272
  ! CHECK-COUNT-3: fir.insert_value
  ! CHECK: fir.call @_FortranAioInputNamelist([[cookie]]
  ! CHECK: fir.call @_FortranAioEndIoStatement([[cookie]]
  read(*, rrr)
end

! CHECK-LABEL: global_pointer
subroutine global_pointer
  real,pointer,save::ptrarray(:)
  ! CHECK: %[[a0:.*]] = fir.address_of
  namelist/mygroup/ptrarray
  ! CHECK: %[[a1:.*]] = fir.convert %[[a0]]
  ! CHECK: %[[a2:.*]] = fir.call @_FortranAioBeginExternalListOutput({{.*}}, %[[a1]], {{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[a3:.*]] = fir.address_of
  ! CHECK: %[[a4:.*]] = fir.convert %[[a3]]
  ! CHECK: %[[a5:.*]] = fir.call @_FortranAioOutputNamelist(%[[a2]], %[[a4]])
  ! CHECK: %[[a6:.*]] = fir.call @_FortranAioEndIoStatement(%[[a2]])
  write(10, nml=mygroup)
end

  ! CHECK-DAG: fir.global linkonce @_QQcl.6A6A6A00 constant : !fir.char<1,4>
  ! CHECK-DAG: fir.global linkonce @_QQcl.63636300 constant : !fir.char<1,4>
  ! CHECK-DAG: fir.global linkonce @_QQcl.6E6E6E00 constant : !fir.char<1,4>
