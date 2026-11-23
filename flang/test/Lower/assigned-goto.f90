! RUN: bbc -emit-fir -o - %s | FileCheck %s

  ! CHECK-LABEL: func @_QPnolist
  subroutine nolist
    integer L, V
 11 V = 1
    ! CHECK: fir.store %c31{{.*}} to %{{.}}
    assign 31 to L
    ! CHECK: fir.select %{{.}} : i32 [31, ^bb{{.}}, unit, ^bb{{.}}]
    ! CHECK: fir.call @_FortranAReportFatalUserError
    goto L ! no list
 21 V = 2
    go to 41
 31 V = 3
 41 print*, 3, V
 end

 ! CHECK-LABEL: func @_QPlist
 subroutine list
    integer L, L1, V
 66 format("Nonsense")
    assign 66 to L
    assign 42 to L1
    ! CHECK: fir.store %c22{{.*}} to %{{.}}
    assign 22 to L
 12 V = 100
    ! CHECK: fir.store %c32{{.*}} to %{{.}}
    assign 32 to L
    ! CHECK: fir.select %{{.}} : i32 [22, ^bb{{.}}, 32, ^bb{{.}}, unit, ^bb{{.}}]
    ! CHECK: fir.call @_FortranAReportFatalUserError
    goto L (42, 32, 22, 32, 32) ! duplicate labels are allowed
 22 V = 200
    go to 42
 32 V = 300
 42 print*, 300, V
 end

 subroutine allocated
    integer, allocatable :: L
    integer :: V
 13 V = 1
    allocate(L)
    ! CHECK: %[[N0:.+]] = fir.box_addr %{{.+}}
    ! CHECK: fir.store %c31{{.*}} to %[[N0]]
    assign 31 to L
    ! CHECK: %[[N1:.+]] = fir.box_addr %{{.+}}
    ! CHECK: %[[N2:.+]] = fir.load %[[N1]]
    ! CHECK: fir.select %[[N2]] : i32 [31, ^bb{{.}}, unit, ^bb{{.}}]
    ! CHECK: fir.call @_FortranAReportFatalUserError
    goto L
 23 V = 2
    goto 41
 31 V = 3
 41 print*, 3, V
 end subroutine allocated

    call nolist
    call list
    call allocated
 end
