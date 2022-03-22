! RUN: bbc %s -o - | tco | FileCheck %s
! Test from Fortran source through to LLVM IR.
! UNSUPPORTED: system-windows

! Assumed size array of assumed length character.
program test
  character :: x(3) = (/ '1','2','3' /)
  call sub(x)
contains
  subroutine sub(x)
    character(*) x(:)
    forall (i=1:2)
       x(i:i)(1:1) = x(i+1:i+1)(1:1)
    end forall
    print *,x
  end subroutine sub
end program test

! CHECK-LABEL: define void @_QFPsub({
! CHECK-SAME: , [1 x [3 x i64]] }* %[[arg:.*]])
! CHECK: %[[extent:.*]] = getelementptr { {{.*}}, [1 x [3 x i64]] }, { {{.*}}, [1 x [3 x i64]] }* %[[arg]], i32 0, i32 7, i64 0, i32 1
! CHECK: %[[extval:.*]] = load i64, i64* %[[extent]]
! CHECK: %[[elesize:.*]] = getelementptr { {{.*}}, [1 x [3 x i64]] }, { {{.*}}, [1 x [3 x i64]] }* %[[arg]], i32 0, i32 1
! CHECK: %[[esval:.*]] = load i64, i64* %[[elesize]]
! CHECK: %[[mul:.*]] = mul i64 1, %[[esval]]
! CHECK: %[[mul2:.*]] = mul i64 %[[mul]], %[[extval]], !dbg !18
! CHECK: %[[buff:.*]] = call i8* @malloc(i64 %[[mul2]])
! CHECK: %[[to:.*]] = getelementptr i8, i8* %[[buff]], i64 %
! CHECK: call void @llvm.memmove.p0i8.p0i8.i64(i8* %[[to]], i8* %{{.*}}, i64 %{{.*}}, i1 false)
! CHECK: call void @free(i8* %[[buff]])
! CHECK: call i8* @_FortranAioBeginExternalListOutput
