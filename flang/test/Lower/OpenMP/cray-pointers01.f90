! Test lowering of Cray pointee references.
! RUN: flang -fc1 -emit-hlfir -fopenmp %s -o - 2>&1 | FileCheck %s

module test_host_assoc_cray_pointer
  ! CHECK-LABEL: fir.global @_QMtest_host_assoc_cray_pointerEivar : i64
  real*8 var(*)
  ! CHECK-LABEL: fir.global  @_QMtest_host_assoc_cray_pointerEvar : !fir.array<?xf64>
  pointer(ivar,var)

contains

  ! CHECK-LABEL: func.func @_QMtest_host_assoc_cray_pointerPset_cray_pointer()
  subroutine set_cray_pointer
    ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf64>>>
    ! CHECK: %[[IVAR_ADDR:.*]] = fir.address_of(@_QMtest_host_assoc_cray_pointerEivar) : !fir.ref<i64>
    ! CHECK: %[[IVAR_DECL:.*]]:2 = hlfir.declare %[[IVAR_ADDR]] {uniq_name = "_QMtest_host_assoc_cray_pointerEivar"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
    ! CHECK: %[[VAR_DECL:.*]]:2 = hlfir.declare %[[ALLOCA]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtest_host_assoc_cray_pointerEvar"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)
    real*8 pointee(2)
    pointee(1) = 42.0

    ivar = loc(pointee)

    !$omp parallel default(none) shared(ivar)
    ! CHECK: omp.parallel
    ! CHECK: %[[I_01:.*]] = fir.convert %[[IVAR_DECL]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
    ! CHECK: %[[I_02:.*]] = fir.load %[[I_01]] : !fir.ref<!fir.ptr<i64>>
    ! CHECK: %[[I_03:.*]] = fir.convert %[[VAR_DECL]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<none>>
    ! CHECK: %[[I_04:.*]] = fir.convert %[[I_02]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
    ! CHECK: fir.call @_FortranAPointerAssociateScalar(%[[I_03]], %[[I_04]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
    print *, var(1)
    !$omp end parallel
  end subroutine
end module

program test_cray_pointers_01
  real*8 :: var(*)
  ! CHECK: %[[BOX_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf64>>>
  ! CHECK: %[[IVAR_ALLOCA:.*]] = fir.alloca i64 {bindc_name = "ivar", uniq_name = "_QFEivar"}
  ! CHECK: %[[IVAR_DECL_01:.*]]:2 = hlfir.declare %[[IVAR_ALLOCA]] {uniq_name = "_QFEivar"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  pointer(ivar,var)
  ! CHECK: %[[VAR_DECL_02:.*]]:2 = hlfir.declare %[[BOX_ALLOCA]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEvar"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf64>>>>)

  real*8 pointee(2)
  pointee(1) = 42.0

  !$omp parallel default(none) private(ivar) shared(pointee)
    ! CHECK: omp.parallel private({{.*}} %[[IVAR_DECL_01]]#0 -> %[[ARG0:.*]] : !fir.ref<i64>) {
    ! CHECK:   %[[IVAR_DECL_02:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFEivar"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
    ! CHECK:   hlfir.assign %{{.*}} to %[[IVAR_DECL_02]]#0 : i64, !fir.ref<i64>
    ivar = loc(pointee)
    ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
    ! CHECK:   %[[CONST_2:.*]] = arith.constant 2 : i32
    ! CHECK:   {{.*}} = math.fpowi {{.*}}, %[[CONST_2]] fastmath<contract> : f64, i32
    ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
    var(1) = var(1) ** 2
    ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
    print *, var(1)
    ! CHECK:   omp.terminator
    ! CHECK: }
  !$omp end parallel
end program test_cray_pointers_01
