! This test checks lowering of OpenMP Threadprivate Directive.
! Test for common block.

!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

module test
  integer:: a
  real :: b(2)
  complex, pointer :: c, d(:)
  character(5) :: e, f(2)
  common /blk/ a, b, c, d, e, f

  !$omp threadprivate(/blk/)

!CHECK: fir.global common @_QBblk(dense<0> : vector<103xi8>) : !fir.array<103xi8>

contains
  subroutine sub()
!CHECK:  [[ADDR0:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<103xi8>>
!CHECK:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<103xi8>> -> !fir.ref<!fir.array<103xi8>>
!CHECK-DAG:  [[ADDR1:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
!CHECK-DAG:  [[ADDR2:%.*]] = fir.coordinate_of [[ADDR1]], [[C0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:  [[ADDR3:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK-DAG:  [[ADDR4:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:  [[C1:%.*]] = arith.constant 4 : index
!CHECK-DAG:  [[ADDR5:%.*]] = fir.coordinate_of [[ADDR4]], [[C1]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:  [[ADDR6:%.*]] = fir.convert [[ADDR5]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>
!CHECK-DAG:  [[ADDR7:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:  [[C2:%.*]] = arith.constant 16 : index
!CHECK-DAG:  [[ADDR8:%.*]] = fir.coordinate_of [[ADDR7]], [[C2]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:  [[ADDR9:%.*]] = fir.convert [[ADDR8]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK-DAG:  [[ADDR10:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:  [[C3:%.*]] = arith.constant 40 : index
!CHECK-DAG:  [[ADDR11:%.*]] = fir.coordinate_of [[ADDR10]], [[C3]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:  [[ADDR12:%.*]] = fir.convert [[ADDR11]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!CHECK-DAG:  [[ADDR13:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:  [[C4:%.*]] = arith.constant 88 : index
!CHECK-DAG:  [[ADDR14:%.*]] = fir.coordinate_of [[ADDR13]], [[C4]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:  [[ADDR15:%.*]] = fir.convert [[ADDR14]] : (!fir.ref<i8>) -> !fir.ref<!fir.char<1,5>>
!CHECK-DAG:  [[ADDR16:%.*]] = fir.convert [[NEWADDR0]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:  [[C5:%.*]] = arith.constant 93 : index
!CHECK-DAG:  [[ADDR17:%.*]] = fir.coordinate_of [[ADDR16]], [[C5]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:  [[ADDR18:%.*]] = fir.convert [[ADDR17]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
!CHECK-DAG:  %{{.*}} = fir.load [[ADDR3]] : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.embox [[ADDR6]](%{{.*}}) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
!CHECK-DAG:  %{{.*}} = fir.load [[ADDR9]] : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK-DAG:  %{{.*}} = fir.load [[ADDR12]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!CHECK-DAG:  %{{.*}} = fir.convert [[ADDR15]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.embox [[ADDR18]](%{{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
    print *, a, b, c, d, e, f

    !$omp parallel
!CHECK:    [[ADDR77:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<103xi8>> -> !fir.ref<!fir.array<103xi8>>
!CHECK-DAG:    [[ADDR78:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR79:%.*]] = fir.coordinate_of [[ADDR78]], [[C0:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR80:%.*]] = fir.convert [[ADDR79:%.*]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK-DAG:    [[ADDR81:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR82:%.*]] = fir.coordinate_of [[ADDR81]], [[C1:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR83:%.*]] = fir.convert [[ADDR82:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>
!CHECK-DAG:    [[ADDR84:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR85:%.*]] = fir.coordinate_of [[ADDR84]], [[C2:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR86:%.*]] = fir.convert [[ADDR85:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK-DAG:    [[ADDR87:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR88:%.*]] = fir.coordinate_of [[ADDR87]], [[C3:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR89:%.*]] = fir.convert [[ADDR88:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!CHECK-DAG:    [[ADDR90:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR91:%.*]] = fir.coordinate_of [[ADDR90]], [[C4:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR92:%.*]] = fir.convert [[ADDR91:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.char<1,5>>
!CHECK-DAG:    [[ADDR93:%.*]] = fir.convert [[ADDR77]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR94:%.*]] = fir.coordinate_of [[ADDR93]], [[C5:%.*]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR95:%.*]] = fir.convert [[ADDR94:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR80]] : !fir.ref<i32>
!CHECK-DAG:    %{{.*}} = fir.embox [[ADDR83]](%{{.*}}) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR86]] : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR89]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!CHECK-DAG:    %{{.*}} = fir.convert [[ADDR92]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
!CHECK-DAG:    %{{.*}} = fir.embox [[ADDR95]](%{{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
      print *, a, b, c, d, e, f
    !$omp end parallel

!CHECK-DAG:  %{{.*}} = fir.load [[ADDR3]] : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.embox [[ADDR6]](%{{.*}}) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
!CHECK-DAG:  %{{.*}} = fir.load [[ADDR9]] : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK-DAG:  %{{.*}} = fir.load [[ADDR12]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
!CHECK-DAG:  %{{.*}} = fir.convert [[ADDR15]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.embox [[ADDR18]](%{{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
    print *, a, b, c, d, e, f

  end
end
