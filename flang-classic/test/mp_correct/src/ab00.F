!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Preprocessor conditional compilation.
	program test
#ifdef _OPENMP
	call check(1,1,1)
#else
	call check(0,1,1)
#endif
	end
