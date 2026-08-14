! RUN: %flang_fc1 -emit-llvm -fcoarray %s -o - 2>&1 | FileCheck %s --check-prefix=LLVM

! LLVM: @_QFB1Ekk_coarray_handle = linkonce global %_QM__fortran_builtinsT__builtin_prif_coarray_handle_type zeroinitializer, comdat

! LLVM-LABEL:  @_QQmain()
! LLVM:  call void @_QMprifPprif_deallocate_coarray(ptr @_QFB1Ekk_coarray_handle, ptr null, ptr null, ptr null)


block
integer :: kk
allocatable :: kk(:)[:]
endblock
end

