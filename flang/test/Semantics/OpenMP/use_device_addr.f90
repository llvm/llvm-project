! RUN: %flang -fc1 -fopenmp -fdebug-dump-symbols %s | FileCheck %s
! OpenMP Version 5.1
! 2.14.2 use_device_addr clause
! List item that appears in a use_device_addr clause has corresponding storage
! in the device data environment, references to the list item in the associated
! structured block are converted into references to the corresponding list item.

subroutine omp_target_data
   integer :: a(1024)
   !CHECK: b, TARGET size=4096 offset=4096: ObjectEntity type: INTEGER(4) shape: 1_8:1024_8
   integer, target :: b(1024)
   a = 1
   !$omp target data map(tofrom: a) use_device_addr(b)
   !CHECK: b (OmpUseDeviceAddr): HostAssoc
      b = a
   !$omp end target data
end subroutine omp_target_data
