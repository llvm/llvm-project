!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-device %s -o - | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

program test_link

  integer :: test_int = 1
  !$omp declare target link(test_int)

  integer :: test_array_1d(3) = (/1,2,3/)
  !$omp declare target link(test_array_1d)

  integer, pointer :: test_ptr1
  !$omp declare target link(test_ptr1)

  integer, target :: test_target = 1
  !$omp declare target link(test_target)

  integer, pointer :: test_ptr2
  !$omp declare target link(test_ptr2)

  !CHECK-DAG: {{%.*}} = omp.map_info var_ptr({{%.*}} : !fir.ref<i32>, i32) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<i32> {name = "test_int"}
  !$omp target
    test_int = test_int + 1
  !$omp end target


  !CHECK-DAG: {{%.*}} = omp.map_info var_ptr({{%.*}} : !fir.ref<!fir.array<3xi32>>, !fir.array<3xi32>) map_clauses(implicit, tofrom) capture(ByRef) bounds({{%.*}}) -> !fir.ref<!fir.array<3xi32>> {name = "test_array_1d"}
  !$omp target
    do i = 1,3
      test_array_1d(i) = i * 2
    end do
  !$omp end target

  allocate(test_ptr1)
  test_ptr1 = 1
  !CHECK-DAG: {{%.*}} = omp.map_info var_ptr({{%.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(implicit, tofrom) capture(ByRef) members({{%.*}} : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "test_ptr1"}
  !$omp target
    test_ptr1 = test_ptr1 + 1
  !$omp end target

  !CHECK-DAG: {{%.*}} = omp.map_info var_ptr({{%.*}} : !fir.ref<i32>, i32) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<i32> {name = "test_target"}
  !$omp target
    test_target = test_target + 1
  !$omp end target


  !CHECK-DAG: {{%.*}} = omp.map_info var_ptr({{%.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(implicit, tofrom) capture(ByRef) members({{%.*}} : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "test_ptr2"}
  test_ptr2 => test_target
  !$omp target
    test_ptr2 = test_ptr2 + 1
  !$omp end target

end
