
!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!  Test attibute oriented variable initializations
!

program wa05
!
 PARAMETER (N=64)
 INTEGER, dimension(N) :: result, expect

 data expect / &
 ! int2, 
     1, &
 ! int3, 
     6, &
 ! int4, 
     3, &
 ! int5, 
     0, &
 ! int6, 
      12, &
 ! int7, 
      42, &
 ! int8, 
     1, &
 ! int9, 
      21, &
 ! int10, 
     3, &
 ! intArr1, 
     2, 3, 4, 5,  12, &
 ! intArr2, 
     6, 6, 6, 6, 6, &
 ! intArr3, 
     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, & 
 ! intArr4, 
     4, 1, &
 ! t1_param
     1, &
 ! t1_inst1, 
      23, &
 ! t1_inst2, 
     6, &
 ! t1_array1, 
     1, 2, 3, &
 ! t1_array2, 
     1, 1, 1, &
 ! t2inst1, 
     1, 1, 2, 3, 4, 5, &
 ! t2inst2, 
     6, 1, 2, 3, 4, 5, &
 ! t3inst1, 
     6,  42, &
 ! intAry5
     1,2,3,4,5,6,7,8,9,10 /

 parameter(int1 = 6)				
 INTEGER, parameter :: intparamarr(5) = 6	
 INTEGER :: int2				
 INTEGER :: int3				
 INTEGER :: int4				
 INTEGER :: int5				
 INTEGER :: int6				
 INTEGER :: int7				
 INTEGER :: int8				
 INTEGER :: int9				
 INTEGER :: int10				
 data int2, int3, int4, int5 /1,int1,3,0/		
 INTEGER :: intArr1(5)				
 data int8,intArr1 / 1,2,3,4,5,12 /		
 INTEGER :: intArr2(5)				
 data int9,intArr2,int6 /21,int1,int1,int1,int1,int1,12/
 INTEGER :: intArr3(10) 			
 data intArr3 / 10*int1 /			
 INTEGER :: intArr4(2)	 			
 data int10,intArr4 /3.12,4,1.1/		
 INTEGER :: intArr5(10) 			
 data (intArr5(i),i=1,5) / 1,2,3,4,5 /			
 data (intArr5(i),i=6,10) / 6,7,8,9,10 /			
!
 type t1					
    integer :: i				
 end type					
!
 type(t1), parameter :: t1_param = t1(1)	
 type (t1) :: t1_inst1				
 data  t1_inst1, int7 / t1(23),42/		
 type (t1) :: t1_inst2 				
 data  t1_inst2  / t1(int1) /			
!
 type (t1) :: t1_array1(3) 			
 data  t1_array1 /t1(1), t1(2), t1(3)/		
 type (t1) :: t1_array2(3) 			
 data  t1_array2 /3*t1(1)/			
!
 type t2					
   integer :: i					
   integer :: iary(1:5)				
 end type					
!
 type (t2) :: t2inst1				
 data  t2inst1  / t2(1, (/1,2,3,4,5/)) /	
 type (t2) :: t2inst2 				
 data t2inst2  /t2(int1, (/(i,i=1,5)/) ) /	
!
 type t3					
    integer :: i				
    type(t1) :: t1_inst				
 end type					
!
 type(t3) :: t3inst1				
 data t3inst1  / t3(int1, t1(42)) /		
!


   result(1) = int2					
   result(2) = int3					
   result(3) = int4					
   result(4) = int5					
   result(5) = int6					
   result(6) = int7					
   result(7) = int8					
   result(8) = int9					
   result(9) = int10					
   result(10:14) = intArr1				
   result(15:19) = intArr2				
   result(20:29) = intArr3				
   result(30:31) = intArr4				
   result(32) = t1_param%i
   result(33) = t1_inst1%i
   result(34) = t1_inst2%i				
   result(35) = t1_array1(1)%i
   result(36) = t1_array1(2)%i
   result(37) = t1_array1(3)%i
   result(38) = t1_array2(1)%i
   result(39) = t1_array2(2)%i
   result(40) = t1_array2(3)%i
   result(41) = t2inst1%i
   result(42) = t2inst1%iary(1)
   result(43) = t2inst1%iary(2)
   result(44) = t2inst1%iary(3)
   result(45) = t2inst1%iary(4)
   result(46) = t2inst1%iary(5)
   result(47) = t2inst2%i
   result(48) = t2inst2%iary(1)
   result(49) = t2inst2%iary(2)
   result(50) = t2inst2%iary(3)
   result(51) = t2inst2%iary(4)
   result(52) = t2inst2%iary(5)
   result(53) = t3inst1%i
   result(54) = t3inst1%t1_inst%i
   result(55:64) = intArr5				

   call check(result, expect, N);
end program
