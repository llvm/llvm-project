! Test lowering of Cray pointee references.
! RUN: bbc -emit-hlfir -fopenmp %s -o - 2>&1 | FileCheck %s

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
