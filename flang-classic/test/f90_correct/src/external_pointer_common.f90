! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test different combinations of common + external + pointer
program comm
 common /com/ Q1
 external Q1
 pointer Q1

 common /com/ Q2
 pointer Q2
 external Q2

 ! This is a bug, uncomment when fixed
 !external Q3
 !common /com/ Q3
 !pointer Q3

 external Q4
 pointer Q4
 common /com/ Q4

 pointer Q5
 common /com/ Q5
 external Q5

 pointer Q6
 external Q6
 common /com/ Q6
end
