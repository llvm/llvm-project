! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
program p
   integer, parameter :: n = 256
   type t1
      integer :: x(256)
   end type t1

   !$omp declare mapper(xx : t1 :: nn) map(to: nn, nn%x)
   !$omp declare mapper(t1 :: nn) map(from: nn)

   !CHECK-LABEL: omp.declare_mapper @_QQFt1.default : !fir.type<_QFTt1{x:!fir.array<256xi32>}>
   !CHECK-LABEL: omp.declare_mapper @_QQFxx : !fir.type<_QFTt1{x:!fir.array<256xi32>}>

   type(t1) :: a, b
   !CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}, {{.*}}) mapper(@_QQFxx) map_clauses(tofrom) capture(ByRef) -> {{.*}} {name = "a"}
   !CHECK: %[[MAP_B:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}, {{.*}}) mapper(@_QQFt1.default) map_clauses(tofrom) capture(ByRef) -> {{.*}} {name = "b"}
   !CHECK: omp.target map_entries(%[[MAP_A]] -> %{{.*}}, %[[MAP_B]] -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : {{.*}}, {{.*}}, {{.*}}, {{.*}}) {
   !$omp target map(mapper(xx) : a) map(mapper(default) : b)
   do i = 1, n
      b%x(i) = a%x(i)
   end do
   !$omp end target
end program p
