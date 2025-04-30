** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

**---   O format, complex array

	parameter(N=4)
	character*22 buf(2)
	data buf /'3535353535336767676767', '1111111111122222222222'/

	complex c(2)
	integer result(N), expect(N)
	equivalence(result, c)

	read(buf,99) c
99	format(2o11)

	data expect /'35353535353'o, '36767676767'o,
     +  '11111111111'o, '22222222222'o /
	call check(result, expect, N)
	end
