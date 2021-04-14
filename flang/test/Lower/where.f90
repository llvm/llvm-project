! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL @_QQmain

   real :: a(10), b(10)

   ! CHECK-DAG: %[[a:.*]] = fir.address_of(@_QEa) : !fir.ref<!fir.array<10xf32>>
   ! CHECK-DAG: %[[b:.*]] = fir.address_of(@_QEb) : !fir.ref<!fir.array<10xf32>>
   ! CHECK-DAG: fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
   ! CHECK-DAG: fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
   ! CHECK: %[[tvec:.*]] = fir.allocmem !fir.array<10x!fir.logical
   ! CHECK-DAG: fir.array_load %[[tvec]]
   ! CHECK-DAG: %[[four:.*]] = constant 4.0{{.*}} : f32
   ! CHECK: fir.do_loop
   ! CHECK: fir.cmpf "ogt", %{{.*}}, %[[four]]
   ! CHECK: fir.array_merge_store %{{.*}}, %{{.*}} to %[[tvec]]
   ! CHECK: fir.do_loop
   ! CHECK: fir.coordinate_of %[[tvec]]
   ! CHECK: fir.if
   ! CHECK: fir.array_fetch
   ! CHECK: fir.negf
   ! CHECK: fir.array_update
   ! CHECK: } else {
   ! CHECK: }
   ! CHECK: }
   ! CHECK: fir.array_merge_store
   ! CHECK: fir.freemem %[[tvec]]
   where (a > 4.0) b = -a

   ! Test that the basic structure is correct
   where (a > 100.0)
   ! loop with an if
     ! CHECK: %[[tvec:.*]] = fir.allocmem !fir.array<10x!fir.logical
     ! CHECK: fir.array_load
     ! CHECK: fir.array_load
     ! CHECK: %[[cst:.*]] = constant 1.0{{.*}}2 : f32
     ! CHECK: fir.do_loop
     ! CHECK: fir.cmpf "ogt", %{{.*}}, %[[cst]]
     ! CHECK: fir.array_merge_store %{{.*}}, %{{.*}} to %[[tvec]]
     ! CHECK: fir.do_loop
     ! CHECK: fir.coordinate_of %[[tvec]]
     ! CHECK: fir.if
     ! CHECK: fir.array_fetch
     ! CHECK: mulf
     ! CHECK: fir.array_update
     ! CHECK: } else {
     ! CHECK: }
     ! CHECK: }
     ! CHECK: fir.array_merge_store
     b = 2.0 * a
   elsewhere (a > 50.0)
   ! loop with else if
     ! CHECK: %[[uvec:.*]] = fir.allocmem !fir.array<10x!fir.logical
     ! CHECK: fir.array_load
     ! CHECK: fir.array_load
     ! CHECK: %[[cst50:.*]] = constant 5.0{{.*}}1 : f32
     ! CHECK: fir.do_loop
     ! CHECK: fir.cmpf "ogt", %{{.*}}, %[[cst50]]
     ! CHECK: fir.array_merge_store %{{.*}}, %{{.*}} to %[[uvec]]
     ! CHECK: fir.do_loop
     ! CHECK: fir.coordinate_of %[[tvec]]
     ! CHECK: fir.if
     ! CHECK: } else {
     ! CHECK: fir.coordinate_of %[[uvec]]
     ! CHECK: fir.if
     ! CHECK: fir.array_fetch
     ! CHECK: addf
     ! CHECK: fir.array_update
     ! CHECK: } else {
     ! CHECK: }
     ! CHECK: }
     ! CHECK: fir.array_merge_store
     b = 3.0 + a
   ! Use cached conditions
     ! CHECK: fir.do_loop
     ! CHECK: fir.coordinate_of %[[tvec]]
     ! CHECK: fir.if
     ! CHECK: } else {
     ! CHECK: fir.coordinate_of %[[uvec]]
     ! CHECK: fir.if
     ! CHECK: fir.array_fetch
     ! CHECK: subf
     ! CHECK: fir.array_update
     ! CHECK: } else {
     ! CHECK: }
     ! CHECK: }
     ! CHECK: fir.array_merge_store
     a = a - 1.0
   elsewhere
   ! Use cached conditions, always false
     ! CHECK: fir.do_loop
     ! CHECK: fir.coordinate_of %[[tvec]]
     ! CHECK: fir.if
     ! CHECK: } else {
     ! CHECK: fir.coordinate_of %[[uvec]]
     ! CHECK: fir.if
     ! CHECK: } else {
     ! CHECK: fir.array_fetch
     ! CHECK: divf
     ! CHECK: fir.array_update
     ! CHECK: }
     ! CHECK: }
     ! CHECK: fir.array_merge_store
     a = a / 2.0
   end where
   ! CHECK-DAG: freemem %[[tvec]]
   ! CHECK-DAG: freemem %[[uvec]]
   ! CHECK: return
end
