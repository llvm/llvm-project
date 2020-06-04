! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPm
function m(index)
  ! CHECK: fir.select %{{.}} : i32 [1, ^bb{{.}}, 2, ^bb{{.}}, 3, ^bb{{.}}, 4, ^bb{{.}}, 5, ^bb{{.}}, unit, ^bb{{.}}]
  goto (9,7,5,3,1) index ! + 1
  m = 0; return
1 m = 1; return
3 m = 3; return
5 m = 5; return
7 m = 7; return
9 m = 9; return
end

! print*, m(-3); print*, m(0)
! print*, m(1); print*, m(2); print*, m(3); print*, m(4); print*, m(5)
! print*, m(6); print*, m(9)
end
