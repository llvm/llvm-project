!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

! Regression test for https://github.com/llvm/llvm-project/issues/106297

program bug
  implicit none
  integer :: table(10)
  !$OMP PARALLEL PRIVATE(table)
    table = 50
    if (any(table/=50)) then
      stop 'fail 3'
    end if
  !$OMP END PARALLEL
  print *,'ok'
End Program


! CHECK-LABEL: define internal void {{.*}}..omp_par(
! CHECK:       omp.par.entry:
! CHECK:         %[[VAL_9:.*]] = alloca i32, align 4
! CHECK:         %[[VAL_10:.*]] = load i32, ptr %[[VAL_11:.*]], align 4
! CHECK:         store i32 %[[VAL_10]], ptr %[[VAL_9]], align 4
! CHECK:         %[[VAL_12:.*]] = load i32, ptr %[[VAL_9]], align 4
! CHECK:         %[[PRIV_BOX_ALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
! ...
! check that we use the private copy of table for the assignment
! CHECK:       omp.par.region1:
! CHECK:         %[[ELEMENTAL_TMP:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
! CHECK:         %[[TABLE_BOX_ADDR:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
! CHECK:         %[[BOXED_FIFTY:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, align 8
! CHECK:         %[[FIFTY:.*]] = alloca i32, i64 1, align 4
! CHECK:         %[[INTERMEDIATE:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
! CHECK:         %[[TABLE_BOX_ADDR2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i64 1, align 8
! CHECK:         %[[LOAD_26:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[PRIV_BOX_ALLOC]], align 8
! CHECK:         store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[LOAD_26]], ptr %[[INTERMEDIATE]], align 8
! CHECK:         store i32 50, ptr %[[FIFTY]], align 4
! CHECK:         %[[FIFTY_BOX_VAL:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8 } { ptr undef, i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i32 20240719, i8 0, i8 9, i8 0, i8 0 }, ptr %[[FIFTY]], 0
! CHECK:         store { ptr, i64, i32, i8, i8, i8, i8 } %[[FIFTY_BOX_VAL]], ptr %[[BOXED_FIFTY]], align 8
! CHECK:         %[[TABLE_BOX_VAL2:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INTERMEDIATE]], align 8
! CHECK:         store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[TABLE_BOX_VAL2]], ptr %[[TABLE_BOX_ADDR2]], align 8
! CHECK:         call void @_FortranAAssign(ptr %[[TABLE_BOX_ADDR2]], ptr %[[BOXED_FIFTY]], ptr @{{.*}}, i32 9)
! CHECK:         %[[LOAD_29:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[PRIV_BOX_ALLOC]], align 8
! CHECK:         store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[LOAD_29]], ptr %[[TABLE_BOX_ADDR]], align 8
! CHECK:         %[[PRIV_TABLE:.*]] = call ptr @malloc(i64 ptrtoint (ptr getelementptr ([10 x i32], ptr null, i32 1) to i64))
! ...
! check that we use the private copy of table for table/=50
! CHECK:       omp.par.region3:
! CHECK:         %[[VAL_44:.*]] = sub nsw i64 %{{.*}}, 1
! CHECK:         %[[VAL_45:.*]] = mul nsw i64 %[[VAL_44]], 1
! CHECK:         %[[VAL_46:.*]] = mul nsw i64 %[[VAL_45]], 1
! CHECK:         %[[VAL_47:.*]] = add nsw i64 %[[VAL_46]], 0
! CHECK:         %[[VAL_48:.*]] = getelementptr i32, ptr %[[PRIV_TABLE]], i64 %[[VAL_47]]
