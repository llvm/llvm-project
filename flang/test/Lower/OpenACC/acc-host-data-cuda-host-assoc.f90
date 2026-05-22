! RUN: bbc -fopenacc -fcuda -emit-hlfir %s -o - | FileCheck %s

interface something 
  subroutine proc_device(x) 
    real(4), device :: x(100) 
  end subroutine 
  subroutine proc_host(x) 
    real(4) :: x(100) 
  end subroutine 
end interface 

real(4) :: a(100)  
!$acc declare copy(a)  

call test_simple() 
contains  
  subroutine test_simple  
    !$acc host_data use_device(a)  
    call something(a)  
    !$acc end host_data  
  end subroutine  
end 

! CHECK: fir.call @_QPproc_device
