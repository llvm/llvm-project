; Directed test cases for log
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func=log op1=7ff80000.00000001 result=7ff80000.00000001 errno=0
func=log op1=fff80000.00000001 result=7ff80000.00000001 errno=0
func=log op1=7ff00000.00000001 result=7ff80000.00000001 errno=0 status=i
func=log op1=fff00000.00000001 result=7ff80000.00000001 errno=0 status=i
func=log op1=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=log op1=fff00000.00000000 result=7ff80000.00000001 errno=EDOM status=i
func=log op1=7fefffff.ffffffff result=40862e42.fefa39ef.354 errno=0
func=log op1=ffefffff.ffffffff result=7ff80000.00000001 errno=EDOM status=i
func=log op1=3ff00000.00000000 result=00000000.00000000 errno=0
func=log op1=bff00000.00000000 result=7ff80000.00000001 errno=EDOM status=i
func=log op1=00000000.00000000 result=fff00000.00000000 errno=ERANGE status=z
func=log op1=80000000.00000000 result=fff00000.00000000 errno=ERANGE status=z
func=log op1=00000000.00000001 result=c0874385.446d71c3.639 errno=0
func=log op1=80000000.00000001 result=7ff80000.00000001 errno=EDOM status=i
func=log op1=40000000.00000000 result=3fe62e42.fefa39ef.358 errno=0
func=log op1=3fe00000.00000000 result=bfe62e42.fefa39ef.358 errno=0
