! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols %s | FileCheck %s
! OpenMP Version 5.0
! 2.10.1 use_device_ptr clause
! List items that appear in a use_device_ptr clause are converted into device
! pointers to the corresponding list item in the device data environment.

subroutine omp_target_data
   use iso_c_binding
   integer :: a(1024)
   !CHECK: b size=8 offset=4096: ObjectEntity type: TYPE(c_ptr)
   type(C_PTR) :: b
   integer, pointer :: arrayB
   a = 1
   !$omp target data map(tofrom: a, arrayB) use_device_ptr(b)
   !CHECK: b (OmpUseDevicePtr): HostAssoc
      allocate(arrayB)
      call c_f_pointer(b, arrayB)
      a = arrayB
   !$omp end target data
end subroutine omp_target_data

