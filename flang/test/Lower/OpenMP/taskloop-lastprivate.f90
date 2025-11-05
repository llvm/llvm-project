! Test the lastprivate clause when used with the taskloop construct
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - 2>&1 | FileCheck %s

! CHECK-LABEL: omp.private
! CHECK-SAME: {type = private} @[[I_PRIVATE:.*]] : i32
! CHECK-LABEL: omp.private
! CHECK-SAME: {type = private} @[[LAST_I_PRIVATE:.*]] : i32

! CHECK: %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlastprivateEi"}
! CHECK: %[[DECLARE_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFlastprivateEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[ALLOCA_LAST_I:.*]] = fir.alloca i32 {bindc_name = "last_i", uniq_name = "_QFlastprivateElast_i"}
! CHECK: %[[DECLARE_LAST_I:.*]]:2 = hlfir.declare %[[ALLOCA_LAST_I]] {uniq_name = "_QFlastprivateElast_i"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

subroutine lastprivate()
   integer :: i, last_i

   ! CHECK: omp.taskloop
   ! CHECK-SAME: private(@[[LAST_I_PRIVATE]] %[[DECLARE_LAST_I]]#0 -> %arg0, @[[I_PRIVATE]] %[[DECLARE_I]]#0 -> %arg1 : !fir.ref<i32>, !fir.ref<i32>) {
   !$omp taskloop lastprivate(last_i)
   do i=1,10
       last_i = i
   end do
   !$omp end taskloop
end