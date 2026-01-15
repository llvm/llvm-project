! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPi
subroutine i
  implicit none
  integer :: ctemp(10) = (/1,2,3,4,5,6,7,8,9,22/)
  print *, ctemp(1:10)
end subroutine i

! CHECK-LABEL: func @_QPs
subroutine s
  integer, parameter :: LONGreal = 8
  real (kind = LONGreal), dimension(-1:11) :: x = (/0,0,0,0,0,0,0,0,0,0,0,0,0/)
  real (kind = LONGreal), dimension(0:12) :: g = (/0,0,0,0,0,0,0,0,0,0,0,0,0/)
  real (kind = LONGreal) :: gs(13)
  x(1) = 4.0
  g(1) = 5.0
  ! CHECK: %[[g_decl:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsEg"}
  ! CHECK: %[[gs_decl:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsEgs"}
  ! CHECK: %[[x_decl:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsEx"}
  ! CHECK: %[[g_slice:.*]] = hlfir.designate %[[g_decl]]#0 (%c0{{.*}}:%c12{{.*}}:%c1{{.*}})
  ! CHECK: %[[x_slice:.*]] = hlfir.designate %[[x_decl]]#0 (%c11{{.*}}:%c-1{{.*}}:%c-1{{.*}})
  ! CHECK: %[[res:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<13xf64> {
  ! CHECK: ^bb0(%[[idx:.*]]: index):
  ! CHECK:   %[[g_elem:.*]] = hlfir.designate %[[g_slice]] (%[[idx]])
  ! CHECK:   %[[x_elem:.*]] = hlfir.designate %[[x_slice]] (%[[idx]])
  ! CHECK:   %[[g_val:.*]] = fir.load %[[g_elem]]
  ! CHECK:   %[[x_val:.*]] = fir.load %[[x_elem]]
  ! CHECK:   %[[sum:.*]] = arith.addf %[[g_val]], %[[x_val]]
  ! CHECK:   hlfir.yield_element %[[sum]]
  ! CHECK: }
  ! CHECK: hlfir.assign %[[res]] to %[[gs_decl]]#0
  gs = g(0:12:1) + x(11:(-1):(-1))
  print *, gs
  !print *, dot_product(g(0:12:1), x(11:(-1):(-1)))
end subroutine s

! CHECK-LABEL: func @_QPs2
subroutine s2
  real :: x(10)
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFs2Ex"}
  x = 0.0
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  print *, x
  ! CHECK: %[[slice:.*]] = hlfir.designate %[[x]]#0 (%c1{{.*}}:%c10{{.*}}:%c3{{.*}})
  ! CHECK: hlfir.assign %cst{{.*}} to %[[slice]]
  x(1:10:3) = 2.0
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  print *, x
end subroutine s2

! CHECK-LABEL: func @_QQmain
program main
  integer :: A(10)
  A(1) = 1
  A(2) = 2
  A(3) = 3
  print *, A
  ! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFEa"}
  ! CHECK: %[[slice:.*]] = hlfir.designate %[[A]]#0 (%c1{{.*}}:%c3{{.*}}:%c1{{.*}})
  ! CHECK: %[[box:.*]] = fir.embox %[[slice]]
  ! CHECK: fir.convert %[[box]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
  print*, A(1:3:1)
  call s
  call i
end program main
