! RUN: bbc -emit-hlfir -mlir-print-debuginfo -o - %s | FileCheck %s

  integer :: n = 0, x = 1
  if (x .ne. 1) goto 9
  n = n + 1
  if (x .gt. 1) goto 9
  n = n + 1
9 print *, 'n =', n
end

! CHECK-LABEL: c.func @_QQmain
  ! CHECK:  %[[V_0:[0-9]+]] = fir.address_of(@_QFEn) : !fir.ref<i32> loc("{{.*}}if-loc.f90":3:
  ! CHECK:  %[[V_1:[0-9]+]]:2 = hlfir.declare %[[V_0]] {uniq_name = "_QFEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>) loc("{{.*}}if-loc.f90":3:
  ! CHECK:  %[[V_2:[0-9]+]] = fir.address_of(@_QFEx) : !fir.ref<i32> loc("{{.*}}if-loc.f90":3:
  ! CHECK:  %[[V_3:[0-9]+]]:2 = hlfir.declare %[[V_2]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>) loc("{{.*}}if-loc.f90":3:
  ! CHECK:  %[[V_4:[0-9]+]] = fir.load %[[V_3]]#0 : !fir.ref<i32> loc("{{.*}}if-loc.f90":4:
  ! CHECK:  %[[V_5:[0-9]+]] = arith.cmpi ne, %[[V_4]], %c1{{.*}} : i32 loc("{{.*}}if-loc.f90":4:
  ! CHECK:  %[[V_6:[0-9]+]] = arith.xori %[[V_5]], %true{{[_0-9]*}} : i1 loc("{{.*}}if-loc.f90":4:
  ! CHECK:  fir.if %[[V_6]] {
  ! CHECK:    %[[V_18:[0-9]+]] = fir.load %[[V_1]]#0 : !fir.ref<i32> loc("{{.*}}if-loc.f90":5:
  ! CHECK:    %[[V_19:[0-9]+]] = arith.addi %[[V_18]], %c1{{.*}} : i32 loc("{{.*}}if-loc.f90":5:
  ! CHECK:    hlfir.assign %[[V_19]] to %[[V_1]]#0 : i32, !fir.ref<i32> loc("{{.*}}if-loc.f90":5:
  ! CHECK:    %[[V_20:[0-9]+]] = fir.load %[[V_3]]#0 : !fir.ref<i32> loc("{{.*}}if-loc.f90":6:
  ! CHECK:    %[[V_21:[0-9]+]] = arith.cmpi sgt, %[[V_20]], %c1{{.*}} : i32 loc("{{.*}}if-loc.f90":6:
  ! CHECK:    %[[V_22:[0-9]+]] = arith.xori %[[V_21]], %true{{[_0-9]*}} : i1 loc("{{.*}}if-loc.f90":6:
  ! CHECK:    fir.if %[[V_22]] {
  ! CHECK:      %[[V_23:[0-9]+]] = fir.load %[[V_1]]#0 : !fir.ref<i32> loc("{{.*}}if-loc.f90":7:
  ! CHECK:      %[[V_24:[0-9]+]] = arith.addi %[[V_23]], %c1{{.*}} : i32 loc("{{.*}}if-loc.f90":7:
  ! CHECK:      hlfir.assign %[[V_24]] to %[[V_1]]#0 : i32, !fir.ref<i32> loc("{{.*}}if-loc.f90":7:
  ! CHECK:    }
  ! CHECK:  }
  ! CHECK:  %[[V_9:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput{{.*}} loc("{{.*}}if-loc.f90":8:
  ! CHECK:  return loc("{{.*}}if-loc.f90":9:
  ! CHECK:}
