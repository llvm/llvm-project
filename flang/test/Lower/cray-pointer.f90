! RUN: bbc %s -emit-fir -hlfir=false -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! Test Cray Pointers

! Test Scalar Case

! CHECK-LABEL: func.func @_QPcray_scalar() {
subroutine cray_scalar()
  integer :: i, pte
  integer :: data = 3
  integer :: j = -3
  pointer(ptr, pte)
  ptr = loc(data)

! CHECK: %[[data:.*]] = fir.address_of(@_QFcray_scalarEdata) {{.*}}
! CHECK: %[[i:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[j:.*]] = fir.address_of(@_QFcray_scalarEj) {{.*}}
! CHECK: %[[ptr:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[databox:.*]] = fir.embox %[[data]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[dataaddr:.*]] = fir.box_addr %[[databox]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[dataaddrval:.*]] = fir.convert %[[dataaddr]] : (!fir.ref<i32>) -> i64
! CHECK: fir.store %[[dataaddrval]] to %[[ptr]] : !fir.ref<i64>

  i = pte
  print *, i

! CHECK: %[[ptrbox:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<i64>
! CHECK: %[[ptraddr:.*]] = fir.box_addr %[[ptrbox]] : (!fir.box<i64>) -> !fir.ref<i64>
! CHECK: %[[ptraddrval:.*]] = fir.convert %[[ptraddr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptraddrval]] : !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptrldd:.*]] = fir.load %[[ptrld]] : !fir.ptr<i32>
! CHECK: fir.store %[[ptrldd]] to %[[i]] : !fir.ref<i32>

  pte = j
  print *, data, pte

! CHECK: %[[jld:.*]] = fir.load %[[j]] : !fir.ref<i32>
! CHECK: %[[ptrbox1:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<i64>
! CHECK: %[[ptraddr1:.*]] = fir.box_addr %[[ptrbox1]] : (!fir.box<i64>) -> !fir.ref<i64>
! CHECK: %[[ptraddrval1:.*]] = fir.convert %[[ptraddr1]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptrld1:.*]] = fir.load %[[ptraddrval1]] : !fir.ref<!fir.ptr<i32>>
! CHECK: fir.store %[[jld]] to %[[ptrld1]] : !fir.ptr<i32>

end

! Test Derived Type Case

! CHECK-LABEL: func.func @_QPcray_derivedtype() {
subroutine cray_derivedType()
  integer :: pte, k
  type dt
    integer :: i, j
  end type
  type(dt) :: xdt
  pointer(ptr, pte)
  xdt = dt(-1, -3)
  ptr = loc(xdt)

! CHECK: %[[dt:.*]] = fir.alloca !fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>
! CHECK: %[[k:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[pte:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[ptr:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[xdt:.*]] = fir.alloca !fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}> {{.*}}
! CHECK: %[[xdtbox:.*]] = fir.embox %[[xdt]] : (!fir.ref<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>) -> !fir.box<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>
! CHECK: %[[xdtaddr:.*]] = fir.box_addr %[[xdtbox]] : (!fir.box<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>) -> !fir.ref<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>
! CHECK: %[[xdtaddrval:.*]] = fir.convert %[[xdtaddr]] : (!fir.ref<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>) -> i64
! CHECK: fir.store %[[xdtaddrval]] to %[[ptr]] : !fir.ref<i64>

  k = pte
  print *, k

! CHECK: %[[ptrbox:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<i64>
! CHECK: %[[ptraddr:.*]] = fir.box_addr %[[ptrbox]] : (!fir.box<i64>) -> !fir.ref<i64>
! CHECK: %[[ptraddrval:.*]] = fir.convert %[[ptraddr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptraddrval]] : !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptrldd:.*]] = fir.load %[[ptrld]] : !fir.ptr<i32>
! CHECK: fir.store %[[ptrldd]] to %[[k]] : !fir.ref<i32>

  pte = k + 2
  print *, xdt, pte

! CHECK: %[[kld:.*]] = fir.load %[[k]] : !fir.ref<i32>
! CHECK: %[[kld1:.*]] = fir.load %[[k]] : !fir.ref<i32>
! CHECK: %[[const:.*]] = arith.constant 2 : i32
! CHECK: %[[add:.*]] = arith.addi %[[kld1]], %[[const]] : i32
! CHECK: %[[ptrbox1:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<i64>
! CHECK: %[[ptraddr1:.*]] = fir.box_addr %[[ptrbox1]] : (!fir.box<i64>) -> !fir.ref<i64>
! CHECK: %[[ptraddrval1:.*]] = fir.convert %[[ptraddr1]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptrld1:.*]] = fir.load %[[ptraddrval1]] : !fir.ref<!fir.ptr<i32>>
! CHECK: fir.store %[[add]] to %[[ptrld1]] : !fir.ptr<i32>

end

! Test Ptr arithmetic Case

! CHECK-LABEL: func.func @_QPcray_ptrarth() {
subroutine cray_ptrArth()
  integer :: pte, i
  pointer(ptr, pte)
  type dt
    integer :: x, y, z
  end type
  type(dt) :: xdt
  xdt = dt(5, 11, 2)
  ptr = loc(xdt)

! CHECK: %[[dt:.*]] = fir.alloca !fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>
! CHECK: %[[i:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[pte:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[ptr:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[xdt:.*]] = fir.alloca !fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}> {{.*}}
! CHECK: %[[xdtbox:.*]] = fir.embox %[[xdt]] : (!fir.ref<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>) -> !fir.box<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>
! CHECK: %[[xdtaddr:.*]] = fir.box_addr %[[xdtbox]] : (!fir.box<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>
! CHECK: %[[xdtaddrval:.*]] = fir.convert %[[xdtaddr]] : (!fir.ref<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>) -> i64
! CHECK: fir.store %[[xdtaddrval]] to %[[ptr]] : !fir.ref<i64>

  ptr = ptr + 4
  i = pte
  print *, i

! CHECK: %[[ptrbox:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<i64>
! CHECK: %[[ptraddr:.*]] = fir.box_addr %[[ptrbox]] : (!fir.box<i64>) -> !fir.ref<i64>
! CHECK: %[[ptraddrval:.*]] = fir.convert %[[ptraddr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptraddrval]] : !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptrldd:.*]] = fir.load %[[ptrld]] : !fir.ptr<i32>
! CHECK: fir.store %[[ptrldd]] to %[[i]] : !fir.ref<i32>

  ptr = ptr + 4
  pte = -7
  print *, xdt

! CHECK: %[[ld:.*]] = fir.load %[[ptr]] : !fir.ref<i64>
! CHECK: %[[const:.*]] = arith.constant 4 : i64
! CHECK: %[[add:.*]] = arith.addi %[[ld]], %[[const]] : i64
! CHECK: fir.store %[[add]] to %[[ptr]] : !fir.ref<i64>
! CHECK: %[[const1:.*]] = arith.constant -7 : i32
! CHECK: %[[ptrbox1:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<i64>
! CHECK: %[[ptraddr1:.*]] = fir.box_addr %[[ptrbox1]] : (!fir.box<i64>) -> !fir.ref<i64>
! CHECK: %[[ptraddrval1:.*]] = fir.convert %[[ptraddr1]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptrld1:.*]] = fir.load %[[ptraddrval1]] : !fir.ref<!fir.ptr<i32>>
! CHECK: fir.store %[[const1]] to %[[ptrld1]] : !fir.ptr<i32>

end

! Test Array element Case

! CHECK-LABEL: func.func @_QPcray_arrayelement() {
subroutine cray_arrayElement()
  integer :: pte, k, data(5)
  pointer (ptr, pte(3))
  data = [ 1, 2, 3, 4, 5 ]
  ptr = loc(data(2))

! CHECK: %[[data:.*]] = fir.alloca !fir.array<5xi32> {{.*}}
! CHECK: %[[k:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[pte:.*]] = fir.alloca !fir.array<3xi32> {{.*}}
! CHECK: %[[ptr:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[c2:.*]] = arith.constant 2 : i64
! CHECK: %[[c1:.*]] = arith.constant 1 : i64
! CHECK: %[[sub:.*]] = arith.subi %[[c2]], %[[c1]] : i64
! CHECK: %[[cor:.*]] = fir.coordinate_of %[[data]], %[[sub]] : (!fir.ref<!fir.array<5xi32>>, i64) -> !fir.ref<i32>
! CHECK: %[[box:.*]] = fir.embox %[[cor]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i32>) -> i64
! CHECK: fir.store %[[val]] to %[[ptr]] : !fir.ref<i64>

  k = pte(3)
  print *, k

! CHECK: %[[c3:.*]] = arith.constant 3 : i64
! CHECK: %[[c1:.*]] = arith.constant 1 : i64
! CHECK: %[[sub:.*]] = arith.subi %[[c3]], %[[c1]] : i64
! CHECK: %[[box:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<!fir.ref<i64>>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ref<i64>>) -> !fir.ref<i64>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[ld1:.*]] = fir.load %[[val]] : !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[cor:.*]] = fir.coordinate_of %[[ld1]], %[[sub]] : (!fir.ptr<!fir.array<3xi32>>, i64) -> !fir.ref<i32>
! CHECK: %[[ld2:.*]] = fir.load %[[cor]] : !fir.ref<i32>
! CHECK: fir.store %[[ld2]] to %[[k]] : !fir.ref<i32>

  pte(2) = -2
  print *, data

! CHECK: %[[c2n:.*]] = arith.constant -2 : i32
! CHECK: %[[c2:.*]] = arith.constant 2 : i64
! CHECK: %[[c1:.*]] = arith.constant 1 : i64
! CHECK: %[[sub:.*]] = arith.subi %[[c2]], %[[c1]] : i64
! CHECK: %[[box:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<!fir.ref<i64>>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ref<i64>>) -> !fir.ref<i64>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[ld1:.*]] = fir.load %[[val]] : !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[cor:.*]] = fir.coordinate_of %[[ld1]], %[[sub]] : (!fir.ptr<!fir.array<3xi32>>, i64) -> !fir.ref<i32>
! CHECK: fir.store %[[c2n]] to %[[cor]] : !fir.ref<i32>

end

! Test 2d Array element Case

! CHECK-LABEL: func.func @_QPcray_2darrayelement() {
subroutine cray_2darrayElement()
  integer :: pte, k, data(2,4)
  pointer (ptr, pte(2,3))
  data = reshape([1,2,3,4,5,6,7,8], [2,4])
  ptr = loc(data(2,2))

! CHECK: %[[data:.*]] = fir.alloca !fir.array<2x4xi32> {{.*}}
! CHECK: %[[k:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[pte:.*]] = fir.alloca !fir.array<2x3xi32> {{.*}}
! CHECK: %[[ptr:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[c2:.*]] = arith.constant 2 : i64
! CHECK: %[[c1:.*]] = arith.constant 1 : i64
! CHECK: %[[sub1:.*]] = arith.subi %[[c2]], %[[c1]] : i64
! CHECK: %[[c22:.*]] = arith.constant 2 : i64
! CHECK: %[[c12:.*]] = arith.constant 1 : i64
! CHECK: %[[sub2:.*]] = arith.subi %[[c22]], %[[c12]] : i64
! CHECK: %[[cor:.*]] = fir.coordinate_of %[[data]], %[[sub1]], %[[sub2]] : (!fir.ref<!fir.array<2x4xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK: %[[box:.*]] = fir.embox %[[cor]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i32>) -> i64
! CHECK: fir.store %[[val]] to %[[ptr]] : !fir.ref<i64>

  k = pte(1,1)
  print *, k

! CHECK: %[[c2:.*]] = arith.constant 1 : i64
! CHECK: %[[c1:.*]] = arith.constant 1 : i64
! CHECK: %[[sub1:.*]] = arith.subi %[[c2]], %[[c1]] : i64
! CHECK: %[[c22:.*]] = arith.constant 1 : i64
! CHECK: %[[c12:.*]] = arith.constant 1 : i64
! CHECK: %[[sub2:.*]] = arith.subi %[[c22]], %[[c12]] : i64
! CHECK: %[[box:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<!fir.ref<i64>>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ref<i64>>) -> !fir.ref<i64>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<!fir.array<2x3xi32>>>
! CHECK: %[[ld1:.*]] = fir.load %[[val]] : !fir.ref<!fir.ptr<!fir.array<2x3xi32>>>
! CHECK: %[[cor:.*]] = fir.coordinate_of %[[ld1]], %[[sub1]], %[[sub2]] : (!fir.ptr<!fir.array<2x3xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK: %[[ld2:.*]] = fir.load %[[cor]] : !fir.ref<i32>
! CHECK: fir.store %[[ld2]] to %[[k]] : !fir.ref<i32>

  pte(1,2) = -2
  print *, data

! CHECK: %[[c2n:.*]] = arith.constant -2 : i32
! CHECK: %[[c2:.*]] = arith.constant 1 : i64
! CHECK: %[[c1:.*]] = arith.constant 1 : i64
! CHECK: %[[sub1:.*]] = arith.subi %[[c2]], %[[c1]] : i64
! CHECK: %[[c22:.*]] = arith.constant 2 : i64
! CHECK: %[[c12:.*]] = arith.constant 1 : i64
! CHECK: %[[sub2:.*]] = arith.subi %[[c22]], %[[c12]] : i64
! CHECK: %[[box:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<!fir.ref<i64>>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ref<i64>>) -> !fir.ref<i64>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<!fir.array<2x3xi32>>>
! CHECK: %[[ld1:.*]] = fir.load %[[val]] : !fir.ref<!fir.ptr<!fir.array<2x3xi32>>>
! CHECK: %[[cor:.*]] = fir.coordinate_of %[[ld1]], %[[sub1]], %[[sub2]] : (!fir.ptr<!fir.array<2x3xi32>>, i64, i64) -> !fir.ref<i32>
! CHECK: fir.store %[[c2n]] to %[[cor]] : !fir.ref<i32>

end

! Test Whole Array case

! CHECK-LABEL: func.func @_QPcray_array() {
subroutine cray_array()
  integer :: pte, k(3), data(5)
  pointer (ptr, pte(3))
  data = [ 1, 2, 3, 4, 5 ]
  ptr = loc(data(2))

! CHECK: %[[data:.*]] = fir.alloca !fir.array<5xi32> {{.*}}
! CHECK: %[[c3:.*]] = arith.constant 3 : index
! CHECK: %[[k:.*]] = fir.alloca !fir.array<3xi32> {{.*}}
! CHECK: %[[c31:.*]] = arith.constant 3 : index
! CHECK: %[[pte:.*]] = fir.alloca !fir.array<3xi32> {{.*}}
! CHECK: %[[ptr:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[c2:.*]] = arith.constant 2 : i64
! CHECK: %[[c1:.*]] = arith.constant 1 : i64
! CHECK: %[[sub:.*]] = arith.subi %[[c2]], %[[c1]] : i64
! CHECK: %[[cor:.*]] = fir.coordinate_of %[[data]], %[[sub]] : (!fir.ref<!fir.array<5xi32>>, i64) -> !fir.ref<i32>
! CHECK: %[[box:.*]] = fir.embox %[[cor]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i32>) -> i64
! CHECK: fir.store %[[val]] to %[[ptr]] : !fir.ref<i64>

  k = pte
  print *, k

! CHECK: %[[shape1:.*]] = fir.shape %[[c3]] : (index) -> !fir.shape<1>
! CHECK: %[[arrayld1:.*]] = fir.array_load %[[k]](%[[shape1]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK: %[[shape:.*]] = fir.shape %[[c31]] : (index) -> !fir.shape<1>
! CHECK: %[[box:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<!fir.ref<i64>>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ref<i64>>) -> !fir.ref<i64>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[ld:.*]] =  fir.load %[[val]] : !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[arrayld:.*]] = fir.array_load %[[ld]](%[[shape]]) : (!fir.ptr<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK: %[[c1:.*]] = arith.constant 1 : index
! CHECK: %[[c0:.*]] = arith.constant 0 : index
! CHECK: %[[sub:.*]] = arith.subi %[[c3]], %[[c1]] : index
! CHECK: %[[doloop:.*]] = fir.do_loop %arg0 = %[[c0]] to %[[sub]] step %[[c1]] unordered iter_args(%arg1 = %[[arrayld1]]) -> (!fir.array<3xi32>) {
! CHECK: %[[arrayfetch:.*]] = fir.array_fetch %[[arrayld]], %arg0 : (!fir.array<3xi32>, index) -> i32
! CHECK: %[[arrayupdate:.*]] = fir.array_update %arg1, %[[arrayfetch]], %arg0 : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
! CHECK: fir.result %[[arrayupdate]] : !fir.array<3xi32>
! CHECK: fir.array_merge_store %[[arrayld1]], %[[doloop]] to %[[k]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>

  pte = -2
  print *, data

! CHECK: %[[shape:.*]] = fir.shape %[[c31]] : (index) -> !fir.shape<1>
! CHECK: %[[box:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<!fir.ref<i64>>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ref<i64>>) -> !fir.ref<i64>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[ld:.*]] = fir.load %[[val]] : !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[arrayld:.*]] = fir.array_load %[[ld]](%[[shape]]) : (!fir.ptr<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK: %[[c2n:.*]] = arith.constant -2 : i32
! CHECK: %[[c1:.*]] = arith.constant 1 : index
! CHECK: %[[c0:.*]] = arith.constant 0 : index
! CHECK: %[[sub1:.*]] = arith.subi %[[c31]], %[[c1]] : index
! CHECK: %[[doloop:.*]] = fir.do_loop %arg0 = %[[c0]] to %[[sub1]] step %[[c1]] unordered iter_args(%arg1 = %[[arrayld]]) -> (!fir.array<3xi32>) {
! CHECK: %[[arrayupdate:.*]] = fir.array_update %arg1, %[[c2n]], %arg0 : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
! CHECK: fir.result %[[arrayupdate]] : !fir.array<3xi32>
! CHECK: fir.array_merge_store %[[arrayld]], %[[doloop]] to %[[ld]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ptr<!fir.array<3xi32>>
end

! Test Array Section  case

! CHECK-LABEL: func.func @_QPcray_arraysection() {
subroutine cray_arraySection()
  integer :: pte, k(2), data(5)
  pointer (ptr, pte(3))
  data = [ 1, 2, 3, 4, 5 ]
  ptr = loc(data(2))

! CHECK: %[[c5:.*]] = arith.constant 5 : index
! CHECK: %[[data:.*]] = fir.alloca !fir.array<5xi32> {{.*}}
! CHECK: %[[c2:.*]] = arith.constant 2 : index
! CHECK: %[[k:.*]] = fir.alloca !fir.array<2xi32> {{.*}}
! CHECK: %[[c3:.*]] = arith.constant 3 : index
! CHECK: %[[pte:.*]] = fir.alloca !fir.array<3xi32> {{.*}}
! CHECK: %[[ptr:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[c1:.*]] = arith.constant 2 : i64
! CHECK: %[[c0:.*]] = arith.constant 1 : i64
! CHECK: %[[sub:.*]] = arith.subi %[[c1]], %[[c0]] : i64
! CHECK: %[[cor:.*]] = fir.coordinate_of %[[data]], %[[sub]] : (!fir.ref<!fir.array<5xi32>>, i64) -> !fir.ref<i32>
! CHECK: %[[box:.*]] = fir.embox %[[cor]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i32>) -> i64
! CHECK: fir.store %[[val]] to %[[ptr]] : !fir.ref<i64>

  k = pte(2:3)
  print *, k

! CHECK: %[[shape1:.*]] = fir.shape %[[c2]] : (index) -> !fir.shape<1>
! CHECK: %[[arrayld1:.*]] = fir.array_load %[[k]](%[[shape1]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.array<2xi32>
! CHECK: %[[c2i64:.*]] = arith.constant 2 : i64
! CHECK: %[[conv:.*]] = fir.convert %[[c2i64]] : (i64) -> index
! CHECK: %[[c1i64:.*]] = arith.constant 1 : i64
! CHECK: %[[conv1:.*]] = fir.convert %[[c1i64]] : (i64) -> index
! CHECK: %[[c3i64:.*]] = arith.constant 3 : i64
! CHECK: %[[conv2:.*]] = fir.convert %[[c3i64]] : (i64) -> index
! CHECK: %[[shape:.*]] = fir.shape %[[c3]] : (index) -> !fir.shape<1>
! CHECK: %[[slice:.*]] = fir.slice %[[conv]], %[[conv2]], %[[conv1]] : (index, index, index) -> !fir.slice<1>
! CHECK: %[[box:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<!fir.ref<i64>>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ref<i64>>) -> !fir.ref<i64>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[ld:.*]] =  fir.load %[[val]] : !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[arrayld:.*]] = fir.array_load %[[ld]](%[[shape]]) [%[[slice]]] : (!fir.ptr<!fir.array<3xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<3xi32>
! CHECK: %[[c1_3:.*]] = arith.constant 1 : index
! CHECK: %[[c0_4:.*]] = arith.constant 0 : index
! CHECK: %[[sub:.*]] = arith.subi %[[c2]], %[[c1_3]] : index
! CHECK: %[[doloop:.*]] = fir.do_loop %arg0 = %[[c0_4]] to %[[sub]] step %[[c1_3]] unordered iter_args(%arg1 = %[[arrayld1]]) -> (!fir.array<2xi32>) {
! CHECK: %[[arrayfetch:.*]] = fir.array_fetch %[[arrayld]], %arg0 : (!fir.array<3xi32>, index) -> i32
! CHECK: %[[arrayupdate:.*]] = fir.array_update %arg1, %[[arrayfetch]], %arg0 : (!fir.array<2xi32>, i32, index) -> !fir.array<2xi32>
! CHECK: fir.result %[[arrayupdate]] : !fir.array<2xi32>
! CHECK: fir.array_merge_store %[[arrayld1]], %[[doloop]] to %[[k]] : !fir.array<2xi32>, !fir.array<2xi32>, !fir.ref<!fir.array<2xi32>>

  pte(1:2) = -2
  print *, data

! CHECK: %[[c1_5:.*]] = arith.constant 1 : i64
! CHECK: %[[conv:.*]] = fir.convert %[[c1_5]] : (i64) -> index
! CHECK: %[[c1_6:.*]] = arith.constant 1 : i64
! CHECK: %[[conv1:.*]] = fir.convert %[[c1_6]] : (i64) -> index
! CHECK: %[[c2_7:.*]] = arith.constant 2 : i64
! CHECK: %[[conv2:.*]] = fir.convert %[[c2_7]] : (i64) -> index
! CHECK: %[[c0_8:.*]] = arith.constant 0 : index
! CHECK: %[[sub:.*]] = arith.subi %[[conv2]], %[[conv]] : index
! CHECK: %[[add:.*]]  = arith.addi %[[sub]], %[[conv1]] : index
! CHECK: %[[div:.*]] = arith.divsi %[[add]], %[[conv1]] : index
! CHECK: %[[cmp:.*]] = arith.cmpi sgt, %[[div]], %[[c0_8]] : index
! CHECK: %[[sel:.*]] = arith.select %[[cmp]], %[[div]], %[[c0_8]] : index
! CHECK: %[[shape:.*]] = fir.shape %[[c3]] : (index) -> !fir.shape<1>
! CHECK: %[[slice:.*]] = fir.slice %[[conv]], %[[conv2]], %[[conv1]] : (index, index, index) -> !fir.slice<1>
! CHECK: %[[box:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<!fir.ref<i64>>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ref<i64>>) -> !fir.ref<i64>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[ld:.*]] = fir.load %[[val]] : !fir.ref<!fir.ptr<!fir.array<3xi32>>>
! CHECK: %[[arrayld:.*]] = fir.array_load %[[ld]](%[[shape]]) [%[[slice]]] : (!fir.ptr<!fir.array<3xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<3xi32>
! CHECK: %[[c2n:.*]] = arith.constant -2 : i32
! CHECK: %[[c1_9:.*]] = arith.constant 1 : index
! CHECK: %[[c0_8:.*]] = arith.constant 0 : index
! CHECK: %[[sub1:.*]] = arith.subi %[[sel]], %[[c1_9]] : index
! CHECK: %[[doloop:.*]] = fir.do_loop %arg0 = %[[c0_8]] to %[[sub1]] step %[[c1_9]] unordered iter_args(%arg1 = %[[arrayld]]) -> (!fir.array<3xi32>) {
! CHECK: %[[arrayupdate:.*]] = fir.array_update %arg1, %[[c2n]], %arg0 : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
! CHECK: fir.result %[[arrayupdate]] : !fir.array<3xi32>
! CHECK: fir.array_merge_store %[[arrayld]], %[[doloop]] to %[[ld]][%[[slice]]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ptr<!fir.array<3xi32>>, !fir.slice<1>
end

! Test Cray pointer declared in a module
module mod_cray_ptr
  integer :: pte
  pointer(ptr, pte)
end module

! CHECK-LABEL: @_QPtest_ptr
subroutine test_ptr()
  use mod_cray_ptr
  implicit none
  integer :: x
  ptr = loc(x)
! CHECK: %[[ptr:.*]] = fir.address_of(@_QMmod_cray_ptrEptr) : !fir.ref<i64>
! CHECK: %[[x:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_ptrEx"}
! CHECK: %[[box:.*]] = fir.embox %[[x]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[boxAddr:.*]] = fir.box_addr %[[box]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[addr_x:.*]] = fir.convert %[[boxAddr]] : (!fir.ref<i32>) -> i64
! CHECK: fir.store %[[addr_x]] to %[[ptr]] : !fir.ref<i64>
end

subroutine test_pte()
  use mod_cray_ptr
  implicit none
  integer :: x
  pte = x
! CHECK: %[[ptr:.*]] = fir.address_of(@_QMmod_cray_ptrEptr) : !fir.ref<i64>
! CHECK: %[[x:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_pteEx"}
! CHECK: %[[xval:.*]] = fir.load %[[x]] : !fir.ref<i32>
! CHECK: %[[box:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<i64>
! CHECK: %[[boxAddr:.*]] = fir.box_addr %[[box]] : (!fir.box<i64>) -> !fir.ref<i64>
! CHECK: %[[ptr2:.*]] = fir.convert %[[boxAddr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptr2val:.*]] = fir.load %[[ptr2]] : !fir.ref<!fir.ptr<i32>>
! CHECK: fir.store %[[xval]] to %[[ptr2val]] : !fir.ptr<i32>

  x = pte
! CHECK: %[[box2:.*]] = fir.embox %[[ptr]] : (!fir.ref<i64>) -> !fir.box<i64>
! CHECK: %[[box2Addr:.*]] = fir.box_addr %[[box2]] : (!fir.box<i64>) -> !fir.ref<i64>
! CHECK: %[[refptr:.*]] = fir.convert %[[box2Addr]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i32>>
! CHECK: %[[ptr4:.*]] = fir.load %[[refptr]] : !fir.ref<!fir.ptr<i32>>
! CHECK: %[[val:.*]] = fir.load %[[ptr4]] : !fir.ptr<i32>
! CHECK: fir.store %[[val]] to %[[x]] : !fir.ref<i32>
end

