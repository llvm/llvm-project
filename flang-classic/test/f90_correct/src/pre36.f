*
* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*
* This tests # characters in fixed-form fortran code
* # in the initial column should be ignored, and treated as a comment even when
* preprocessing

#if 0
#error "This should never get called"
#endif

	program test
	print *, "Hello"
     #, "World"
	call check(.true., .true., 1)
	END
