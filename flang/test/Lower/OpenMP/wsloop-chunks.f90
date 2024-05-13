! This test checks that chunk size is passed correctly when lowering of
! OpenMP DO Directive(Worksharing) with chunk size

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

program wsloop
        integer :: i
        integer :: chunk

! CHECK-LABEL: func.func @_QQmain() attributes {fir.bindc_name = "wsloop"} {
! CHECK:         %[[CHUNK_REF:.*]] = fir.alloca i32 {bindc_name = "chunk", uniq_name = "_QFEchunk"}
! CHECK:         %[[VAL_0:.*]]:2 = hlfir.declare %[[CHUNK_REF]] {uniq_name = "_QFEchunk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!$OMP DO SCHEDULE(static, 4)

do i=1, 9
  print*, i

! CHECK:         %[[VAL_2:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_3:.*]] = arith.constant 9 : i32
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = arith.constant 4 : i32
! CHECK:         omp.wsloop schedule(static = %[[VAL_5]] : i32) nowait {
! CHECK-NEXT:      omp.loop_nest (%[[ARG0:.*]]) : i32 = (%[[VAL_2]]) to (%[[VAL_3]]) inclusive step (%[[VAL_4]]) {
! CHECK:             fir.store %[[ARG0]] to %[[STORE_IV:.*]]#1 : !fir.ref<i32>
! CHECK:             %[[LOAD_IV:.*]] = fir.load %[[STORE_IV]]#0 : !fir.ref<i32>
! CHECK:             {{.*}} = fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
! CHECK:             omp.yield
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }

end do
!$OMP END DO NOWAIT
!$OMP DO SCHEDULE(static, 2+2)

do i=1, 9
  print*, i*2

! CHECK:         %[[VAL_14:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_15:.*]] = arith.constant 9 : i32
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_17:.*]] = arith.constant 4 : i32
! CHECK:         omp.wsloop schedule(static = %[[VAL_17]] : i32) nowait {
! CHECK-NEXT:      omp.loop_nest (%[[ARG1:.*]]) : i32 = (%[[VAL_14]]) to (%[[VAL_15]]) inclusive step (%[[VAL_16]]) {
! CHECK:             fir.store %[[ARG1]] to %[[STORE_IV1:.*]]#1 : !fir.ref<i32>
! CHECK:             %[[VAL_24:.*]] = arith.constant 2 : i32
! CHECK:             %[[LOAD_IV1:.*]] = fir.load %[[STORE_IV1]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_25:.*]] = arith.muli %[[VAL_24]], %[[LOAD_IV1]] : i32
! CHECK:             {{.*}} = fir.call @_FortranAioOutputInteger32({{.*}}, %[[VAL_25]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
! CHECK:             omp.yield
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
  
end do
!$OMP END DO NOWAIT
chunk = 6
!$OMP DO SCHEDULE(static, chunk)

do i=1, 9
   print*, i*3
end do
!$OMP END DO NOWAIT
! CHECK:         %[[VAL_28:.*]] = arith.constant 6 : i32
! CHECK:         hlfir.assign %[[VAL_28]] to %[[VAL_0]]#0 : i32, !fir.ref<i32>
! CHECK:         %[[VAL_29:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_30:.*]] = arith.constant 9 : i32
! CHECK:         %[[VAL_31:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_32:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<i32>
! CHECK:         omp.wsloop schedule(static = %[[VAL_32]] : i32) nowait {
! CHECK-NEXT:      omp.loop_nest (%[[ARG2:.*]]) : i32 = (%[[VAL_29]]) to (%[[VAL_30]]) inclusive step (%[[VAL_31]]) {
! CHECK:             fir.store %[[ARG2]] to %[[STORE_IV2:.*]]#1 : !fir.ref<i32>
! CHECK:             %[[VAL_39:.*]] = arith.constant 3 : i32
! CHECK:             %[[LOAD_IV2:.*]] = fir.load %[[STORE_IV2]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_40:.*]] = arith.muli %[[VAL_39]], %[[LOAD_IV2]] : i32
! CHECK:             {{.*}} = fir.call @_FortranAioOutputInteger32({{.*}}, %[[VAL_40]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
! CHECK:             omp.yield
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

end
