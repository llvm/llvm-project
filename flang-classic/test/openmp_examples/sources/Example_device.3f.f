! @@name:	device.3f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
program foo
use omp_lib, ONLY : omp_get_default_device, omp_set_default_device
integer :: old_default_device, new_default_device
   old_default_device = omp_get_default_device()
   print*, "Default device = ", old_default_device
   new_default_device = old_default_device + 1
   call omp_set_default_device(new_default_device)
   if (omp_get_default_device() == old_default_device) &
      print*,"Default device is STILL = ", old_default_device
end program
