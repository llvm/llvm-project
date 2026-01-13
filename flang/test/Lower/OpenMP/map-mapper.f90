! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
program p
   integer, parameter :: n = 256
   type t1
      integer :: x(256)
   end type t1

   !$omp declare mapper(xx : t1 :: nn) map(to: nn, nn%x)
   !$omp declare mapper(t1 :: nn) map(from: nn)

   !CHECK-LABEL: omp.declare_mapper @_QQFt1_omp_default_mapper : !fir.type<_QFTt1{x:!fir.array<256xi32>}>
   !CHECK-LABEL: omp.declare_mapper @_QQFxx : !fir.type<_QFTt1{x:!fir.array<256xi32>}>

   type(t1) :: a, b
   !CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}, {{.*}}) map_clauses(tofrom) capture(ByRef) mapper(@_QQFxx) -> {{.*}} {name = "a"}
   !CHECK: omp.target map_entries(%[[MAP_A]] -> %{{.*}}, %{{.*}} -> %{{.*}} : {{.*}}, {{.*}}) {
   !$omp target map(mapper(xx) : a)
   do i = 1, n
      a%x(i) = i
   end do
   !$omp end target

   !CHECK: %[[MAP_B:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}, {{.*}}) map_clauses(tofrom) capture(ByRef) mapper(@_QQFt1_omp_default_mapper) -> {{.*}} {name = "b"}
   !CHECK: omp.target map_entries(%[[MAP_B]] -> %{{.*}}, %{{.*}} -> %{{.*}} : {{.*}}, {{.*}}) {
   !$omp target map(mapper(default) : b)
   do i = 1, n
      b%x(i) = i
   end do
   !$omp end target
end program p
