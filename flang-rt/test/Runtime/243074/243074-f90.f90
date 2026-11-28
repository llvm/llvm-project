! UNSUPPORTED: offload-cuda
! UNSUPPORTED: system-windows

! Addresses bug: [flang-rt]  Fortran I/O from a C++ static destructor spins forever with the static runtime
! https://github.com/llvm/llvm-project/issues/243074
!
! RUN: %clang -c %S/243074-cpp.cpp -o %t-cpp.o
! RUN: %flang -c %S/243074-f90.f90 -o %t-fortran.o
! RUN: %flang %t-cpp.o %t-fortran.o -lstdc++ -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
!
! CHECK: PASS
!
! Original bug is that the Fortran runtime has shutdown the I/O stack
! at the completion of MAIN()
! The C++ destructor, "Reporter()" [in source file 243074-cpp.cpp]
! then calls Fortran subroutine "report_at_exit()", which subsequently
! tries to write to standard output, thus requiring some parts of the
! runtime to restart.
!
! The original implementation had the destructor ~Lock() in:
!    flang-rt/include/flang-rt/runtime/lock.h 
! trying to destroy and  mutex (Lock) while the mutex was still acquired.
!
! 'man pthread_mutex_destroy' states:
!      pthread_mutex_destroy  destroys a mutex object, freeing the resources it
!      might hold.  The mutex must be unlocked  on  entrance.   In  the  Linux‐
!      Threads  implementation, no resources are associated with mutex objects,
!      thus pthread_mutex_destroy actually does nothing  except  checking  that
!      the mutex is unlocked.

subroutine report_at_exit() bind(c)
  print *, 'Subroutine report_at_exit called from C++ static destructor'
  print *, 'PASS'
end subroutine

program main
  print *, 'main done'
end program
