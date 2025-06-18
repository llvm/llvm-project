! This test checks the lowering and application of default map types for the target enter/exit data constructs and map clauses

!RUN: %flang -fc1 -emit-fir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s --check-prefix=CHECK-52
!RUN: not %flang -fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2<&1| FileCheck %s --check-prefix=CHECK-51

module test
  use omp_lib
  real, allocatable :: A

contains
  subroutine initialize()
  allocate(A)
  !$omp target enter data map(A)
  !CHECK-52: %276 = omp.map.info var_ptr(%2 : !fir.ref<!fir.box<!fir.heap<f32>>>, f32) map_clauses(to) capture(ByRef) var_ptr_ptr(%275 : !fir.llvm_ptr<!fir.ref<f32>>) -> !fir.llvm_ptr<!fir.ref<f32>> {name = ""}
  !CHECK-52: %277 = omp.map.info var_ptr(%2 : !fir.ref<!fir.box<!fir.heap<f32>>>, !fir.box<!fir.heap<f32>>) map_clauses(to) capture(ByRef) members(%276 : [0] : !fir.llvm_ptr<!fir.ref<f32>>) -> !fir.ref<!fir.box<!fir.heap<f32>>> {name = "a"}
  !CHECK-51: to and alloc map types are permitted

  end subroutine initialize

  subroutine finalize()
  !$omp target exit data map(A)
  !CHECK-52: %274 = omp.map.info var_ptr(%2 : !fir.ref<!fir.box<!fir.heap<f32>>>, f32) map_clauses(from) capture(ByRef) var_ptr_ptr(%273 : !fir.llvm_ptr<!fir.ref<f32>>) -> !fir.llvm_ptr<!fir.ref<f32>> {name = ""}
  !CHECK-52: %275 = omp.map.info var_ptr(%2 : !fir.ref<!fir.box<!fir.heap<f32>>>, !fir.box<!fir.heap<f32>>) map_clauses(from) capture(ByRef) members(%274 : [0] : !fir.llvm_ptr<!fir.ref<f32>>) -> !fir.ref<!fir.box<!fir.heap<f32>>> {name = "a"}
  !CHECK-51: from, release and delete map types are permitted
  deallocate(A)
  
  end subroutine finalize
end module test
