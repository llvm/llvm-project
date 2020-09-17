! RUN: bbc %s -o - | tco | FileCheck %s

! CHECK: @_QB = common global [8 x i8] zeroinitializer
! CHECK: @_QBx = global { float, float } { float 1.0{{.*}}, float 2.0{{.*}} }
! CHECK: @_QBy = common global [12 x i8] zeroinitializer

! CHECK-LABEL: _QPs0
subroutine s0
  common // a0, b0

  ! CHECK: call void @_QPs(float* bitcast ([8 x i8]* @_QB to float*), float* bitcast (i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_QB, i32 0, i64 4) to float*))
  call s(a0, b0)
end subroutine s0


! CHECK-LABEL: _QPs1
subroutine s1
  common /x/ a1, b1
  data a1 /1.0/, b1 /2.0/

  ! CHECK: call void @_QPs(float* getelementptr inbounds ({ float, float }, { float, float }* @_QBx, i32 0, i32 0), float* bitcast (i8* getelementptr (i8, i8* bitcast ({ float, float }* @_QBx to i8*), i64 4) to float*))
  call s(a1, b1)
end subroutine s1

! CHECK-LABEL: _QPs2
subroutine s2
  common /y/ a2, b2, c2

  ! CHECK: call void @_QPs(float* bitcast ([12 x i8]* @_QBy to float*), float* bitcast (i8* getelementptr inbounds ([12 x i8], [12 x i8]* @_QBy, i32 0, i64 4) to float*))
  call s(a2, b2)
end subroutine s2
