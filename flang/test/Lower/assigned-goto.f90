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

    call nolist
    call list
 end
