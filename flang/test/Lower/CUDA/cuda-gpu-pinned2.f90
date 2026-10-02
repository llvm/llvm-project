! RUN: bbc -emit-hlfir -gpu=pinned %s -o - | FileCheck %s --check-prefixes=CHECK,NOCUDA
! RUN: bbc -emit-hlfir -gpu=pinned -fcuda %s -o - | FileCheck %s --check-prefixes=CHECK,CUDA

integer, allocatable :: a(:)
allocate(a(10))
deallocate(a)
end

! CHECK-LABEL: func.func @_QQmain()

! NOCUDA-NOT: cuf.allocate
! NOCUDA-NOT: cuf.deallocate

! CUDA: cuf.allocate %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<pinned>} -> i32
! CUDA: cuf.deallocate %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<pinned>} -> i32

