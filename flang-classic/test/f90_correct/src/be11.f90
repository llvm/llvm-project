!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Equivalence in modules

        module xxx
        integer i(2)
        integer n1, n2
        equivalence (i(1),n1), (i(2),n2)
        end module
        use xxx
	integer result(4),expect(4)
	data expect/3,99,3,99/
        n1 = 3
	i(2) = 99
	result(1)=n1
	result(2)=n2
	result(3)=i(1)
	result(4)=i(2)
	call check(result,expect,4)
        end
