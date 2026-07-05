! Test lowering of Cray pointee references.
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QQmain() attributes {fir.bindc_name = "TEST_CRAY_POINTERS_02"}
program test_cray_pointers_02
    implicit none

    ! CHECK:   fir.call @_QPnone_shared() fastmath<contract> : () -> ()
    ! CHECK:   fir.call @_QPnone_private() fastmath<contract> : () -> ()
    ! CHECK:   fir.call @_QPnone_firstprivate() fastmath<contract> : () -> ()
    ! CHECK:   fir.call @_QPprivate_shared() fastmath<contract> : () -> ()
    ! CHECK:   fir.call @_QPprivate_firstprivate() fastmath<contract> : () -> ()
    ! CHECK:   fir.call @_QPfirstprivate_shared() fastmath<contract> : () -> ()
    ! CHECK:   fir.call @_QPfirstprivate_private() fastmath<contract> : () -> ()
    call none_shared()
    call none_private()
    call none_firstprivate()
    call private_shared()
    call private_firstprivate()
    call firstprivate_shared()
    call firstprivate_private()
    ! CHECK:   return
    ! CHECK: }
end program test_cray_pointers_02

! CHECK-LABEL: func.func @_QPnone_shared()
subroutine none_shared()
    implicit none
    integer var(*)
    pointer(ivar,var)
    integer pointee(8)

    pointee(1) = 42
    ivar = loc(pointee)

    !$omp parallel num_threads(1) default(none) shared(ivar)
        ! CHECK: omp.parallel
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   {{.*}} = arith.divsi
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        var(1) = var(1) / 2
        print '(A24,I6)', 'none_shared', var(1)
    !$omp end parallel
    ! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPnone_private()
subroutine none_private()
    implicit none
    integer var(*)
    pointer(ivar,var)
    integer pointee(8)

    pointee(1) = 42
    ivar = loc(pointee)

    !$omp parallel num_threads(1) default(none) private(ivar) shared(pointee)
        ! CHECK: omp.parallel
        ivar = loc(pointee)
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   {{.*}} = arith.addi
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        var(1) = var(1) + 2
        print '(A24,I6)', 'none_private', var(1)
    !$omp end parallel
    ! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPnone_firstprivate()
subroutine none_firstprivate()
    implicit none
    integer var(*)
    pointer(ivar,var)
    integer pointee(8)

    pointee(1) = 42
    ivar = loc(pointee)

    !$omp parallel num_threads(1) default(none) firstprivate(ivar)
        ! CHECK: omp.parallel
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   {{.*}} = arith.muli
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        var(1) = var(1) * 2
        print '(A24,I6)', 'none_firstprivate', var(1)
    !$omp end parallel
    ! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPprivate_shared()
subroutine private_shared()
    implicit none
    integer var(*)
    pointer(ivar,var)
    integer pointee(8)

    pointee(1) = 42
    ivar = loc(pointee)

    !$omp parallel num_threads(1) default(private) shared(ivar)
        ! CHECK: omp.parallel
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   {{.*}} = math.ipowi
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        var(1) = var(1) ** 2
        print '(A24,I6)', 'private_shared', var(1)
    !$omp end parallel
    ! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPprivate_firstprivate()
subroutine private_firstprivate()
    implicit none
    integer var(*)
    pointer(ivar,var)
    integer pointee(8)

    pointee(1) = 42
    ivar = loc(pointee)

    !$omp parallel num_threads(1) default(private) firstprivate(ivar)
        ! CHECK: omp.parallel
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   {{.*}} = arith.subi
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        var(1) = var(1) - 2
        print '(A24,I6)', 'private_firstprivate', var(1)
    !$omp end parallel
    ! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPfirstprivate_shared()
subroutine firstprivate_shared()
    implicit none
    integer var(*)
    pointer(ivar,var)
    integer pointee(8)

    pointee(1) = 42
    ivar = loc(pointee)

    !$omp parallel num_threads(1) default(firstprivate) shared(ivar)
        ! CHECK: omp.parallel
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   {{.*}} = arith.divsi
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        var(1) = var(1) / 2
        print '(A24,I6)', 'firstprivate_shared', var(1)
    !$omp end parallel
    ! CHECK: return
end subroutine

! CHECK-LABEL: func.func @_QPfirstprivate_private()
subroutine firstprivate_private()
    implicit none
    integer var(*)
    pointer(ivar,var)
    integer pointee(8)

    pointee(1) = 42
    ivar = loc(pointee)

    !$omp parallel num_threads(1) default(firstprivate) private(ivar) shared(pointee)
        ! CHECK: omp.parallel
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   {{.*}} = math.ipowi
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ! CHECK:   fir.call @_FortranAPointerAssociateScalar({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
        ivar = loc(pointee)
        var(1) = var(1) ** 2
        print '(A24,I6)', 'firstprivate_private', var(1)
    !$omp end parallel
    ! CHECK: return
end subroutine
