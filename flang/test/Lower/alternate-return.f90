! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPss
subroutine ss(n)
  print*, n
  ! CHECK: return{{$}}
  return
! CHECK-LABEL: func @_QPee
entry ee(n,*)
  print*, n
  ! CHECK: return %{{.}} : index
  return 1
end

! CHECK-LABEL: func @_QQmain
  call ss(7)
  call ee(2, *3)
  print*, 'default'
3 print*, 3

  print*, k(10,20)
  print*, k(15,15)
  print*, k(20,10)
end

! CHECK-LABEL: func @_QPk
function k(n1, n2)
  ! CHECK-NOT: ^bb
  ! CHECK: [[selector:%[0-9]+]] = fir.call @_QPs
  ! CHECK-NEXT: fir.select [[selector]] : index [1, ^[[block1:bb[0-9]+]], 2, ^[[block2:bb[0-9]+]], unit, ^[[blockunit:bb[0-9]+]]
  call s(n1, *5, n2, *7)
  ! CHECK: ^[[blockunit]]: // pred: ^bb0
  k =  0; return;
  ! CHECK: ^[[block1]]: // pred: ^bb0
5 k = -1; return;
  ! CHECK: ^[[block2]]: // pred: ^bb0
7 k =  1; return
end

! CHECK-LABEL: func @_QPs
subroutine s(n1, *, n2, *)
  ! CHECK: [[retval:%[0-9]+]] = fir.alloca index {{{.*}}bindc_name = "s"}
  ! CHECK-COUNT-3: fir.store {{.*}} to [[retval]] : !fir.ref<index>
  if (n1 < n2) return 1
  if (n1 > n2) return 2
  ! CHECK: {{.*}} = fir.load [[retval]] : !fir.ref<index>
  ! CHECK-NEXT: return {{.*}} : index
  return
end
