** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

**----  Z format - leading zeros are insignficant

	character*16 buf
	data buf/'00000000ffffffff'/
	integer b
	read(buf, '(z16)', err=22) b
	goto 33
22	b = 99
33	continue
	call check(b, -1, 1)
	end
