! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test Cray Pointers

! Test Scalar Case

! CHECK-LABEL: func.func @_QPcray_scalar() {
subroutine cray_scalar()
  integer :: i, pte
  integer :: data = 3
  integer :: j = -3
  pointer(ptr, pte)
  ptr = loc(data)

! CHECK: %[[pte_alloc:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK: %[[data_addr:.*]] = fir.address_of(@_QFcray_scalarEdata) {{.*}}
! CHECK: %[[data:.*]]:2 = hlfir.declare %[[data_addr]]
! CHECK: %[[i_alloc:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[i:.*]]:2 = hlfir.declare %[[i_alloc]]
! CHECK: %[[j_addr:.*]] = fir.address_of(@_QFcray_scalarEj) {{.*}}
! CHECK: %[[j:.*]]:2 = hlfir.declare %[[j_addr]]
! CHECK: %[[ptr_alloc:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[ptr:.*]]:2 = hlfir.declare %[[ptr_alloc]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFcray_scalarEptr"}
! CHECK: %[[pte:.*]]:2 = hlfir.declare %[[pte_alloc]] {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFcray_scalarEpte"}
! CHECK: %[[zero:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK: %[[zerobox:.*]] = fir.embox %[[zero]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK: fir.store %[[zerobox]] to %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[databox:.*]] = fir.embox %[[data]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[dataaddr:.*]] = fir.box_addr %[[databox]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[dataaddrval:.*]] = fir.convert %[[dataaddr]] : (!fir.ref<i32>) -> i64
! CHECK: hlfir.assign %[[dataaddrval]] to %[[ptr]]#0 : i64, !fir.ref<i64>

  i = pte
  print *, i

! CHECK: %[[ptrcvt:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptrcvt]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr:.*]] = fir.convert %[[ptrld]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt]], %[[rawptr]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[pteaddr:.*]] = fir.box_addr %[[pteload]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: %[[ptevalue:.*]] = fir.load %[[pteaddr]] : !fir.ptr<i32>
! CHECK: hlfir.assign %[[ptevalue]] to %[[i]]#0 : i32, !fir.ref<i32>

  pte = j
  print *, data, pte

! CHECK: %[[jld:.*]] = fir.load %[[j]]#0 : !fir.ref<i32>
! CHECK: %[[ptrcvt2:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld2:.*]] = fir.load %[[ptrcvt2]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt2:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr2:.*]] = fir.convert %[[ptrld2]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt2]], %[[rawptr2]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload2:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[pteaddr2:.*]] = fir.box_addr %[[pteload2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: hlfir.assign %[[jld]] to %[[pteaddr2]] : i32, !fir.ptr<i32>

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

! CHECK: %[[pte_alloc:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK: %[[k_alloc:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[k:.*]]:2 = hlfir.declare %[[k_alloc]]
! CHECK: %[[ptr_alloc:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[ptr:.*]]:2 = hlfir.declare %[[ptr_alloc]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFcray_derivedtypeEptr"}
! CHECK: %[[xdt_alloc:.*]] = fir.alloca !fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}> {{.*}}
! CHECK: %[[xdt:.*]]:2 = hlfir.declare %[[xdt_alloc]]
! CHECK: %[[pte:.*]]:2 = hlfir.declare %[[pte_alloc]] {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFcray_derivedtypeEpte"}
! CHECK: %[[xdtbox:.*]] = fir.embox %[[xdt]]#0 : (!fir.ref<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>) -> !fir.box<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>
! CHECK: %[[xdtaddr:.*]] = fir.box_addr %[[xdtbox]] : (!fir.box<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>) -> !fir.ref<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>
! CHECK: %[[xdtaddrval:.*]] = fir.convert %[[xdtaddr]] : (!fir.ref<!fir.type<_QFcray_derivedtypeTdt{i:i32,j:i32}>>) -> i64
! CHECK: hlfir.assign %[[xdtaddrval]] to %[[ptr]]#0 : i64, !fir.ref<i64>

  k = pte
  print *, k

! CHECK: %[[ptrcvt:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptrcvt]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr:.*]] = fir.convert %[[ptrld]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt]], %[[rawptr]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[pteaddr:.*]] = fir.box_addr %[[pteload]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: %[[ptevalue:.*]] = fir.load %[[pteaddr]] : !fir.ptr<i32>
! CHECK: hlfir.assign %[[ptevalue]] to %[[k]]#0 : i32, !fir.ref<i32>

  pte = k + 2
  print *, xdt, pte

! CHECK: fir.call @_FortranAioEndIoStatement
! CHECK: %[[kld:.*]] = fir.load %[[k]]#0 : !fir.ref<i32>
! CHECK: %[[const:.*]] = arith.constant 2 : i32
! CHECK: %[[add:.*]] = arith.addi %[[kld]], %[[const]] : i32
! CHECK: %[[ptrcvt2:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld2:.*]] = fir.load %[[ptrcvt2]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt2:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr2:.*]] = fir.convert %[[ptrld2]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt2]], %[[rawptr2]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload2:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[pteaddr2:.*]] = fir.box_addr %[[pteload2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: hlfir.assign %[[add]] to %[[pteaddr2]] : i32, !fir.ptr<i32>

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

! CHECK: %[[pte_alloc:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK: %[[i_alloc:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[i:.*]]:2 = hlfir.declare %[[i_alloc]]
! CHECK: %[[ptr_alloc:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[ptr:.*]]:2 = hlfir.declare %[[ptr_alloc]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFcray_ptrarthEptr"}
! CHECK: %[[xdt_alloc:.*]] = fir.alloca !fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}> {{.*}}
! CHECK: %[[xdt:.*]]:2 = hlfir.declare %[[xdt_alloc]]
! CHECK: %[[pte:.*]]:2 = hlfir.declare %[[pte_alloc]] {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFcray_ptrarthEpte"}
! CHECK: %[[xdtbox:.*]] = fir.embox %[[xdt]]#0 : (!fir.ref<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>) -> !fir.box<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>
! CHECK: %[[xdtaddr:.*]] = fir.box_addr %[[xdtbox]] : (!fir.box<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>
! CHECK: %[[xdtaddrval:.*]] = fir.convert %[[xdtaddr]] : (!fir.ref<!fir.type<_QFcray_ptrarthTdt{x:i32,y:i32,z:i32}>>) -> i64
! CHECK: hlfir.assign %[[xdtaddrval]] to %[[ptr]]#0 : i64, !fir.ref<i64>

  ptr = ptr + 4
  i = pte
  print *, i

! CHECK: %[[ptrld:.*]] = fir.load %[[ptr]]#0 : !fir.ref<i64>
! CHECK: %[[const:.*]] = arith.constant 4 : i64
! CHECK: %[[add:.*]] = arith.addi %[[ptrld]], %[[const]] : i64
! CHECK: hlfir.assign %[[add]] to %[[ptr]]#0 : i64, !fir.ref<i64>
! CHECK: %[[ptrcvt:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld2:.*]] = fir.load %[[ptrcvt]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr:.*]] = fir.convert %[[ptrld2]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt]], %[[rawptr]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[pteaddr:.*]] = fir.box_addr %[[pteload]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: %[[ptevalue:.*]] = fir.load %[[pteaddr]] : !fir.ptr<i32>
! CHECK: hlfir.assign %[[ptevalue]] to %[[i]]#0 : i32, !fir.ref<i32>

  ptr = ptr + 4
  pte = -7
  print *, xdt

! CHECK: %[[ptrld3:.*]] = fir.load %[[ptr]]#0 : !fir.ref<i64>
! CHECK: %[[const2:.*]] = arith.constant 4 : i64
! CHECK: %[[add2:.*]] = arith.addi %[[ptrld3]], %[[const2]] : i64
! CHECK: hlfir.assign %[[add2]] to %[[ptr]]#0 : i64, !fir.ref<i64>
! CHECK: %[[neg7:.*]] = arith.constant -7 : i32
! CHECK: %[[ptrcvt2:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld4:.*]] = fir.load %[[ptrcvt2]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt2:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr2:.*]] = fir.convert %[[ptrld4]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt2]], %[[rawptr2]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload2:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[pteaddr2:.*]] = fir.box_addr %[[pteload2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: hlfir.assign %[[neg7]] to %[[pteaddr2]] : i32, !fir.ptr<i32>

end

! Test Array element Case

! CHECK-LABEL: func.func @_QPcray_arrayelement() {
subroutine cray_arrayElement()
  integer :: pte, k, data(5)
  pointer (ptr, pte(3))
  data = [ 1, 2, 3, 4, 5 ]
  ptr = loc(data(2))

! CHECK: %[[pte_alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK: %[[data_alloc:.*]] = fir.alloca !fir.array<5xi32> {{.*}}
! CHECK: %[[data:.*]]:2 = hlfir.declare %[[data_alloc]]
! CHECK: %[[k_alloc:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[k:.*]]:2 = hlfir.declare %[[k_alloc]]
! CHECK: %[[ptr_alloc:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[ptr:.*]]:2 = hlfir.declare %[[ptr_alloc]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFcray_arrayelementEptr"}
! CHECK: %[[pte:.*]]:2 = hlfir.declare %[[pte_alloc]] {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFcray_arrayelementEpte"}
! CHECK: %[[c2:.*]] = arith.constant 2 : index
! CHECK: %[[elem:.*]] = hlfir.designate %[[data]]#0 (%[[c2]])  : (!fir.ref<!fir.array<5xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[box:.*]] = fir.embox %[[elem]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i32>) -> i64
! CHECK: hlfir.assign %[[val]] to %[[ptr]]#0 : i64, !fir.ref<i64>

  k = pte(3)
  print *, k

! CHECK: %[[ptrcvt:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptrcvt]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr:.*]] = fir.convert %[[ptrld]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt]], %[[rawptr]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[c3:.*]] = arith.constant 3 : index
! CHECK: %[[ptedes:.*]] = hlfir.designate %[[pteload]] (%[[c3]])  : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK: %[[ptevalue:.*]] = fir.load %[[ptedes]] : !fir.ref<i32>
! CHECK: hlfir.assign %[[ptevalue]] to %[[k]]#0 : i32, !fir.ref<i32>

  pte(2) = -2
  print *, data

! CHECK: %[[neg2:.*]] = arith.constant -2 : i32
! CHECK: %[[ptrcvt2:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld2:.*]] = fir.load %[[ptrcvt2]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt2:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr2:.*]] = fir.convert %[[ptrld2]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt2]], %[[rawptr2]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload2:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[c2b:.*]] = arith.constant 2 : index
! CHECK: %[[ptedes2:.*]] = hlfir.designate %[[pteload2]] (%[[c2b]])  : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK: hlfir.assign %[[neg2]] to %[[ptedes2]] : i32, !fir.ref<i32>

end

! Test 2d Array element Case

! CHECK-LABEL: func.func @_QPcray_2darrayelement() {
subroutine cray_2darrayElement()
  integer :: pte, k, data(2,4)
  pointer (ptr, pte(2,3))
  data = reshape([1,2,3,4,5,6,7,8], [2,4])
  ptr = loc(data(2,2))

! CHECK: %[[pte_alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xi32>>>
! CHECK: %[[data_alloc:.*]] = fir.alloca !fir.array<2x4xi32> {{.*}}
! CHECK: %[[data:.*]]:2 = hlfir.declare %[[data_alloc]]
! CHECK: %[[k_alloc:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[k:.*]]:2 = hlfir.declare %[[k_alloc]]
! CHECK: %[[ptr_alloc:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[ptr:.*]]:2 = hlfir.declare %[[ptr_alloc]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFcray_2darrayelementEptr"}
! CHECK: %[[pte:.*]]:2 = hlfir.declare %[[pte_alloc]] {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFcray_2darrayelementEpte"}
! CHECK: %[[elem:.*]] = hlfir.designate %[[data]]#0 (%{{.*}}, %{{.*}})  : (!fir.ref<!fir.array<2x4xi32>>, index, index) -> !fir.ref<i32>
! CHECK: %[[box:.*]] = fir.embox %[[elem]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[val:.*]] = fir.convert %[[addr]] : (!fir.ref<i32>) -> i64
! CHECK: hlfir.assign %[[val]] to %[[ptr]]#0 : i64, !fir.ref<i64>

  k = pte(1,1)
  print *, k

! CHECK: %[[ptrcvt:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptrcvt]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr:.*]] = fir.convert %[[ptrld]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt]], %[[rawptr]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
! CHECK: %[[c1:.*]] = arith.constant 1 : index
! CHECK: %[[c1b:.*]] = arith.constant 1 : index
! CHECK: %[[ptedes:.*]] = hlfir.designate %[[pteload]] (%[[c1]], %[[c1b]])  : (!fir.box<!fir.ptr<!fir.array<?x?xi32>>>, index, index) -> !fir.ref<i32>
! CHECK: %[[ptevalue:.*]] = fir.load %[[ptedes]] : !fir.ref<i32>
! CHECK: hlfir.assign %[[ptevalue]] to %[[k]]#0 : i32, !fir.ref<i32>

  pte(1,2) = -2
  print *, data

! CHECK: %[[neg2:.*]] = arith.constant -2 : i32
! CHECK: %[[ptrcvt2:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld2:.*]] = fir.load %[[ptrcvt2]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt2:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr2:.*]] = fir.convert %[[ptrld2]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt2]], %[[rawptr2]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload2:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
! CHECK: %[[c1c:.*]] = arith.constant 1 : index
! CHECK: %[[c2c:.*]] = arith.constant 2 : index
! CHECK: %[[ptedes2:.*]] = hlfir.designate %[[pteload2]] (%[[c1c]], %[[c2c]])  : (!fir.box<!fir.ptr<!fir.array<?x?xi32>>>, index, index) -> !fir.ref<i32>
! CHECK: hlfir.assign %[[neg2]] to %[[ptedes2]] : i32, !fir.ref<i32>

end

! Test Whole Array case

! CHECK-LABEL: func.func @_QPcray_array() {
subroutine cray_array()
  integer :: pte, k(3), data(5)
  pointer (ptr, pte(3))
  data = [ 1, 2, 3, 4, 5 ]
  ptr = loc(data(2))

! CHECK: %[[pte_alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK: %[[data_alloc:.*]] = fir.alloca !fir.array<5xi32> {{.*}}
! CHECK: %[[data:.*]]:2 = hlfir.declare %[[data_alloc]]
! CHECK: %[[k_alloc:.*]] = fir.alloca !fir.array<3xi32> {{.*}}
! CHECK: %[[k:.*]]:2 = hlfir.declare %[[k_alloc]]
! CHECK: %[[ptr_alloc:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[ptr:.*]]:2 = hlfir.declare %[[ptr_alloc]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFcray_arrayEptr"}
! CHECK: %[[pte:.*]]:2 = hlfir.declare %[[pte_alloc]] {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFcray_arrayEpte"}

  k = pte
  print *, k

! CHECK: %[[ptrcvt:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptrcvt]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr:.*]] = fir.convert %[[ptrld]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt]], %[[rawptr]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: hlfir.assign %[[pteload]] to %[[k]]#0 : !fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.ref<!fir.array<3xi32>>

  pte = -2
  print *, data

! CHECK: %[[neg2:.*]] = arith.constant -2 : i32
! CHECK: %[[ptrcvt2:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld2:.*]] = fir.load %[[ptrcvt2]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt2:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr2:.*]] = fir.convert %[[ptrld2]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt2]], %[[rawptr2]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload2:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: hlfir.assign %[[neg2]] to %[[pteload2]] : i32, !fir.box<!fir.ptr<!fir.array<?xi32>>>
end

! Test Array Section  case

! CHECK-LABEL: func.func @_QPcray_arraysection() {
subroutine cray_arraySection()
  integer :: pte, k(2), data(5)
  pointer (ptr, pte(3))
  data = [ 1, 2, 3, 4, 5 ]
  ptr = loc(data(2))

! CHECK: %[[pte_alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK: %[[data_alloc:.*]] = fir.alloca !fir.array<5xi32> {{.*}}
! CHECK: %[[data:.*]]:2 = hlfir.declare %[[data_alloc]]
! CHECK: %[[k_alloc:.*]] = fir.alloca !fir.array<2xi32> {{.*}}
! CHECK: %[[k:.*]]:2 = hlfir.declare %[[k_alloc]]
! CHECK: %[[ptr_alloc:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[ptr:.*]]:2 = hlfir.declare %[[ptr_alloc]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFcray_arraysectionEptr"}
! CHECK: %[[pte:.*]]:2 = hlfir.declare %[[pte_alloc]] {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFcray_arraysectionEpte"}

  k = pte(2:3)
  print *, k

! CHECK: %[[ptrcvt:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptrcvt]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr:.*]] = fir.convert %[[ptrld]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt]], %[[rawptr]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[ptedes:.*]] = hlfir.designate %[[pteload]] (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<2xi32>>
! CHECK: hlfir.assign %[[ptedes]] to %[[k]]#0 : !fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>

  pte(1:2) = -2
  print *, data

! CHECK: %[[neg2:.*]] = arith.constant -2 : i32
! CHECK: %[[ptrcvt2:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld2:.*]] = fir.load %[[ptrcvt2]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt2:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr2:.*]] = fir.convert %[[ptrld2]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt2]], %[[rawptr2]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload2:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[ptedes2:.*]] = hlfir.designate %[[pteload2]] (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<2xi32>>
! CHECK: hlfir.assign %[[neg2]] to %[[ptedes2]] : i32, !fir.ref<!fir.array<2xi32>>
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
! CHECK: %[[ptr_addr:.*]] = fir.address_of(@_QMmod_cray_ptrEptr) : !fir.ref<i64>
! CHECK: %[[ptr:.*]]:2 = hlfir.declare %[[ptr_addr]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QMmod_cray_ptrEptr"}
! CHECK: %[[x_alloc:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_ptrEx"}
! CHECK: %[[x:.*]]:2 = hlfir.declare %[[x_alloc]]
! CHECK: %[[xbox:.*]] = fir.embox %[[x]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[xboxAddr:.*]] = fir.box_addr %[[xbox]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[addr_x:.*]] = fir.convert %[[xboxAddr]] : (!fir.ref<i32>) -> i64
! CHECK: hlfir.assign %[[addr_x]] to %[[ptr]]#0 : i64, !fir.ref<i64>
end

subroutine test_pte()
  use mod_cray_ptr
  implicit none
  integer :: x
  pte = x
! CHECK-LABEL: func.func @_QPtest_pte()
! CHECK: %[[ptr_addr:.*]] = fir.address_of(@_QMmod_cray_ptrEptr) : !fir.ref<i64>
! CHECK: %[[ptr:.*]]:2 = hlfir.declare %[[ptr_addr]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QMmod_cray_ptrEptr"}
! CHECK: %[[x_alloc:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_pteEx"}
! CHECK: %[[x:.*]]:2 = hlfir.declare %[[x_alloc]]
! CHECK: %[[pte:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QMmod_cray_ptrEpte"}
! CHECK: %[[xval:.*]] = fir.load %[[x]]#0 : !fir.ref<i32>
! CHECK: %[[ptrcvt:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld:.*]] = fir.load %[[ptrcvt]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr:.*]] = fir.convert %[[ptrld]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt]], %[[rawptr]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[pteaddr:.*]] = fir.box_addr %[[pteload]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: hlfir.assign %[[xval]] to %[[pteaddr]] : i32, !fir.ptr<i32>

  x = pte
! CHECK: %[[ptrcvt2:.*]] = fir.convert %[[ptr]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptrld2:.*]] = fir.load %[[ptrcvt2]] : !fir.ref<!fir.ptr<i64>>
! CHECK: %[[ptebox_cvt2:.*]] = fir.convert %[[pte]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[rawptr2:.*]] = fir.convert %[[ptrld2]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[ptebox_cvt2]], %[[rawptr2]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
! CHECK: %[[pteload2:.*]] = fir.load %[[pte]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[pteaddr2:.*]] = fir.box_addr %[[pteload2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: %[[val:.*]] = fir.load %[[pteaddr2]] : !fir.ptr<i32>
! CHECK: hlfir.assign %[[val]] to %[[x]]#0 : i32, !fir.ref<i32>
end
