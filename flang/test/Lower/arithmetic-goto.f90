! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPkagi
function kagi(index)
  ! CHECK: fir.select_case %{{.}} : i32 [#fir.upper, %c-1_i32, ^bb{{.}}, #fir.lower, %c1_i32, ^bb{{.}}, unit, ^bb{{.}}]
  if (index) 7, 8, 9
  kagi = 0; return
7 kagi = 1; return
8 kagi = 2; return
9 kagi = 3; return
end

! CHECK-LABEL: func @_QPkagf
function kagf(findex)
  ! CHECK: %[[zero:.+]] = arith.constant 0.0
  ! CHECK: %{{.+}} = arith.cmpf olt, %{{.+}}, %[[zero]] : f32
  ! CHECK: cond_br %
  ! CHECK: %{{.+}} = arith.cmpf ogt, %{{.+}}, %[[zero]] : f32
  ! CHECK: cond_br %
  ! CHECK: br ^
  if (findex+findex) 7, 8, 9
  kagf = 0; return
7 kagf = 1; return
8 kagf = 2; return
9 kagf = 3; return
end

! CHECK-LABEL: func @_QQmain
  do i = -2, 2
    print*, kagi(i)
  enddo

  print*, kagf(-2.0)
  print*, kagf(-1.0)
  print*, kagf(-0.0)
  print*, kagf( 0.0)
  print*, kagf(+0.0)
  print*, kagf(+1.0)
  print*, kagf(+2.0)
end
