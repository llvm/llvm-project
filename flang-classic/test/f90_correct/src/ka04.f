** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*  incorrectly computing invariancy for array references

	program ka04
	parameter( N = 1 )
	integer result(N), expect(N)
	integer sum
	data m/3/, expect/3/

	result(1) = 0
	j = 1
	do 50 i = 1, m
50		result(1) = result(j) + 1

	call check(result, expect, N)

	end
