C Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
C See https://llvm.org/LICENSE.txt for license information.
C SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
C
C  PACK intrinsic with a scalar mask fails in 6.0 with -mcmodel=medium
C  Failure mode is a runtime error "0: PACK: invalid mask descriptor"
       integer*4 xo(5), xe(5)
       data xe/1, 2, 3, 4, 5/
       xo = pack(xe,.true.)
       call check(xe, xo, 5)
       end
