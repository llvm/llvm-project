! RUN: %flang_fc1 -emit-llvm -fcoarray %s -o - 2>&1 | FileCheck %s --check-prefix=LLVM

! LLVM: @_QFB1Ekk_coarray_handle = linkonce global { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr null, i64 40, i32 20240719, i8 0, i8 42, i8 0, i8 1, ptr @_QMprifEXdtXprif_coarray_handle, [1 x i64] zeroinitializer }, comdat

! LLVM-LABEL:  @_QQmain()
! LLVM:       call void @llvm.memcpy.p0.p0.i32(ptr align 8 %[[VAL_1:.*]], ptr align 8 @_QFB1Ekk_coarray_handle, i32 40, i1 false)
! LLVM-NEXT:  call void @_QMprifPprif_deallocate_coarray(ptr %[[VAL_1]], ptr null, ptr null, ptr null)


block
integer :: kk
allocatable :: kk(:)[:]
endblock
end

