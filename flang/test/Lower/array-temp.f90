! RUN: bbc -emit-hlfir -fwrapv %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPss1()
subroutine ss1
  ! CHECK: %[[shape:[0-9]+]] = fir.shape {{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[aa:[0-9]+]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFss1Eaa"}
  integer, parameter :: N = 2650000
  real aa(N)
  ! CHECK: hlfir.assign %{{.*}} to %[[aa]]#0
  aa = -2
  ! CHECK: %[[slice1:.*]] = hlfir.designate %[[aa]]#0 (%c1{{.*}}:%c2649999{{.*}}:%c1{{.*}})
  ! CHECK: %[[res:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<2649999xf32> {
  ! CHECK:   hlfir.yield_element
  ! CHECK: }
  ! CHECK: %[[slice2:.*]] = hlfir.designate %[[aa]]#0 (%c2{{.*}}:%c2650000{{.*}}:%c1{{.*}})
  ! CHECK: hlfir.assign %[[res]] to %[[slice2]]
  ! CHECK: hlfir.destroy %[[res]]
  aa(2:N) = aa(1:N-1) + 7.0
! print*, aa(1:2), aa(N-1:N)
end

subroutine ss2(N)
  real aa(N)
  aa = -2
  aa(2:N) = aa(1:N-1) + 7.0
  print*, aa(1:2), aa(N-1:N)
end

subroutine ss3(N)
  real aa(2,N)
  aa = -2
  aa(:,2:N) = aa(:,1:N-1) + 7.0
  print*, aa(:,1:2), aa(:,N-1:N)
end

subroutine ss4(N)
  real aa(N,2)
  aa = -2
  aa(2:N,:) = aa(1:N-1,:) + 7.0
  print*, aa(1:2,:), aa(N-1:N,:)
end

! CHECK-LABEL: func @_QPss2(
! CHECK-SAME:               %arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:   %[[aa:[0-9]+]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFss2Eaa"}
! CHECK:   hlfir.assign %{{.*}} to %[[aa]]#0
! CHECK:   %[[slice1:.*]] = hlfir.designate %[[aa]]#0 (%c1{{.*}}:%{{.*}}:%c1{{.*}})
! CHECK:   %[[res:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:     hlfir.designate %[[slice1]]
! CHECK:     fir.load
! CHECK:     arith.addf
! CHECK:     hlfir.yield_element
! CHECK:   }
! CHECK:   %[[slice2:.*]] = hlfir.designate %[[aa]]#0 (%c2{{.*}}:%{{.*}}:%c1{{.*}})
! CHECK:   hlfir.assign %[[res]] to %[[slice2]]
! CHECK:   hlfir.destroy %[[res]]
! CHECK:   fir.call @_FortranAioBeginExternalListOutput
! CHECK:   fir.call @_FortranAioOutputDescriptor
! CHECK:   fir.call @_FortranAioOutputDescriptor
! CHECK:   fir.call @_FortranAioEndIoStatement
! CHECK:   return
! CHECK:   }

! CHECK-LABEL: func @_QPss3(
! CHECK-SAME:               %arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:   %[[aa:[0-9]+]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFss3Eaa"}
! CHECK:   hlfir.assign %{{.*}} to %[[aa]]#0
! CHECK:   %[[slice1:.*]] = hlfir.designate %[[aa]]#0 (%c1{{.*}}:%{{.*}}:%c1{{.*}}, %c1{{.*}}:%{{.*}}:%c1{{.*}})
! CHECK:   %[[res:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<2>) -> !hlfir.expr<2x?xf32> {
! CHECK:     hlfir.designate %[[slice1]]
! CHECK:     fir.load
! CHECK:     arith.addf
! CHECK:     hlfir.yield_element
! CHECK:   }
! CHECK:   %[[slice2:.*]] = hlfir.designate %[[aa]]#0 (%c1{{.*}}:%{{.*}}:%c1{{.*}}, %c2{{.*}}:%{{.*}}:%c1{{.*}})
! CHECK:   hlfir.assign %[[res]] to %[[slice2]]
! CHECK:   hlfir.destroy %[[res]]
! CHECK:   fir.call @_FortranAioBeginExternalListOutput
! CHECK:   fir.call @_FortranAioOutputDescriptor
! CHECK:   fir.call @_FortranAioOutputDescriptor
! CHECK:   fir.call @_FortranAioEndIoStatement
! CHECK:   return
! CHECK:   }

! CHECK-LABEL: func @_QPss4(
! CHECK-SAME:               %arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:   %[[aa:[0-9]+]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFss4Eaa"}
! CHECK:   hlfir.assign %{{.*}} to %[[aa]]#0
! CHECK:   %[[slice1:.*]] = hlfir.designate %[[aa]]#0 (%c1{{.*}}:%{{.*}}:%c1{{.*}}, %c1{{.*}}:%{{.*}}:%c1{{.*}})
! CHECK:   %[[res:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<2>) -> !hlfir.expr<?x2xf32> {
! CHECK:     hlfir.designate %[[slice1]]
! CHECK:     fir.load
! CHECK:     arith.addf
! CHECK:     hlfir.yield_element
! CHECK:   }
! CHECK:   %[[slice2:.*]] = hlfir.designate %[[aa]]#0 (%c2{{.*}}:%{{.*}}:%c1{{.*}}, %c1{{.*}}:%{{.*}}:%c1{{.*}})
! CHECK:   hlfir.assign %[[res]] to %[[slice2]]
! CHECK:   hlfir.destroy %[[res]]
! CHECK:   fir.call @_FortranAioBeginExternalListOutput
! CHECK:   fir.call @_FortranAioOutputDescriptor
! CHECK:   fir.call @_FortranAioOutputDescriptor
! CHECK:   fir.call @_FortranAioEndIoStatement
! CHECK:   return
! CHECK:   }

! CHECK-LABEL: func @_QPtt1
subroutine tt1
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<3xf32>
  ! CHECK: %[[temp_decl:.*]]:2 = hlfir.declare %[[temp]]
  ! CHECK: fir.do_loop %[[arg:.*]] =
  ! CHECK:   %[[const:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.1xr4.0"}
  ! CHECK:   %[[expr:.*]] = hlfir.as_expr %[[const]]#0
  ! CHECK:   %[[assoc:.*]]:3 = hlfir.associate %[[expr]]
  ! CHECK:   %[[box:.*]] = fir.embox %[[assoc]]#0
  ! CHECK:   %[[conv:.*]] = fir.convert %[[box]]
  ! CHECK:   %[[res:.*]] = fir.call @_QFtt1Pr(%[[conv]])
  ! CHECK:   %[[elem:.*]] = hlfir.designate %[[temp_decl]]#0 (%{{.*}})
  ! CHECK:   hlfir.assign %[[res]] to %[[elem]]
  ! CHECK:   hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2
  ! CHECK: }
  ! CHECK: %[[expr2:.*]] = hlfir.as_expr %[[temp_decl]]#0
  ! CHECK: %[[assoc2:.*]]:3 = hlfir.associate %[[expr2]]
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  ! CHECK: hlfir.end_associate %[[assoc2]]#1, %[[assoc2]]#2
  ! CHECK: hlfir.destroy %[[expr2]]
  ! CHECK: fir.call @_FortranAioEndIoStatement
  print*, [(r([7.0]),i=1,3)]
contains
  ! CHECK-LABEL: func private @_QFtt1Pr
  function r(x)
    real x(:)
    r = x(1)
  end
end
