! This test checks lowering of OpenMP Threadprivate Directive.
! Test for character, array, and character array.

!RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s

module test
  character :: x
  integer :: y(5)
  character(5) :: z(5)

  !$omp threadprivate(x, y, z)

!CHECK-DAG: fir.global @_QMtestEx : !fir.char<1> {
!CHECK-DAG: fir.global @_QMtestEy : !fir.array<5xi32> {
!CHECK-DAG: fir.global @_QMtestEz : !fir.array<5x!fir.char<1,5>> {

contains
  subroutine sub()
!CHECK-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QMtestEx) : !fir.ref<!fir.char<1>>
!CHECK-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.char<1>> -> !fir.ref<!fir.char<1>>
!CHECK-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<!fir.array<5xi32>>
!CHECK-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<5xi32>> -> !fir.ref<!fir.array<5xi32>>
!CHECK-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QMtestEz) : !fir.ref<!fir.array<5x!fir.char<1,5>>>
!CHECK-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.array<5x!fir.char<1,5>>> -> !fir.ref<!fir.array<5x!fir.char<1,5>>>
!CHECK-DAG:  %{{.*}} = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.embox [[NEWADDR1]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
!CHECK-DAG:  %{{.*}} = fir.embox [[NEWADDR2]](%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<5x!fir.char<1,5>>>
    print *, x, y, z

    !$omp parallel
!CHECK-DAG:    [[ADDR33:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.char<1>> -> !fir.ref<!fir.char<1>>
!CHECK-DAG:    [[ADDR34:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<5xi32>> -> !fir.ref<!fir.array<5xi32>>
!CHECK-DAG:    [[ADDR35:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.array<5x!fir.char<1,5>>> -> !fir.ref<!fir.array<5x!fir.char<1,5>>>
!CHECK-DAG:    %{{.*}} = fir.convert [[ADDR33]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
!CHECK-DAG:    %{{.*}} = fir.embox [[ADDR34]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
!CHECK-DAG:    %{{.*}} = fir.embox [[ADDR35]](%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<5x!fir.char<1,5>>>
      print *, x, y, z
    !$omp end parallel

!CHECK-DAG:  %{{.*}} = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.embox [[NEWADDR1]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
!CHECK-DAG:  %{{.*}} = fir.embox [[NEWADDR2]](%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<5x!fir.char<1,5>>>
    print *, x, y, z

  end
end
