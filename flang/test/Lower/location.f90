! RUN: bbc -emit-hlfir --mlir-print-debuginfo %s -o - | FileCheck %s

program test
include 'location0.inc'

end 

! CHECK-LABEL: func.func @_QQmain() attributes {fir.bindc_name = "test"} {
! CHECK: fir.call @_FortranAioOutputAscii(%{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1 loc(fused<#fir<loc_kind_array[ base,  inclusion,  inclusion]>>["{{.*}}location1.inc":1:10, "{{.*}}location0.inc":1:1, "{{.*}}location.f90":4:1])
! CHECK: return loc("{{.*}}location.f90":6:1)
! CHECK: } loc("{{.*}}location.f90":3:1)


