! RUN: bbc -emit-hlfir -gpu=pinned -fcuda %s -o - | FileCheck %s

integer, allocatable :: a(:)
integer, allocatable, device :: b(:)
allocate(a(10))
allocate(b(10))
deallocate(a)
deallocate(b)
end

! CHECK-LABEL: func.func @_QQmain()
! CHECK: cuf.allocate %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<pinned>} -> i32
! CHECK: cuf.allocate %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<device>} -> i32
! CHECK: cuf.deallocate %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<pinned>} -> i32
! CHECK: cuf.deallocate %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {data_attr = #cuf.cuda<device>} -> i32

