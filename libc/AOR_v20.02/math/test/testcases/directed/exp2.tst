; Directed test cases for exp2
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func=exp2 op1=7ff80000.00000001 result=7ff80000.00000001 errno=0
func=exp2 op1=fff80000.00000001 result=7ff80000.00000001 errno=0
func=exp2 op1=7ff00000.00000001 result=7ff80000.00000001 errno=0 status=i
func=exp2 op1=fff00000.00000001 result=7ff80000.00000001 errno=0 status=i
func=exp2 op1=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=exp2 op1=fff00000.00000000 result=00000000.00000000 errno=0
func=exp2 op1=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=ox
func=exp2 op1=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=ux
func=exp2 op1=00000000.00000000 result=3ff00000.00000000 errno=0
func=exp2 op1=80000000.00000000 result=3ff00000.00000000 errno=0
func=exp2 op1=00000000.00000001 result=3ff00000.00000000 errno=0
func=exp2 op1=80000000.00000001 result=3ff00000.00000000 errno=0
func=exp2 op1=3ca00000.00000000 result=3ff00000.00000000.58c errno=0
func=exp2 op1=bc900000.00000000 result=3fefffff.ffffffff.a74 errno=0
func=exp2 op1=3fe00000.00000000 result=3ff6a09e.667f3bcc.909 errno=0
func=exp2 op1=bfe00000.00000000 result=3fe6a09e.667f3bcc.909 errno=0
func=exp2 op1=3ff00000.00000000 result=40000000.00000000 errno=0
func=exp2 op1=bff00000.00000000 result=3fe00000.00000000 errno=0
func=exp2 op1=40000000.00000000 result=40100000.00000000 errno=0
func=exp2 op1=c0000000.00000000 result=3fd00000.00000000 errno=0
func=exp2 op1=3ff12345.6789abcd result=4000cef3.c5d12321.663 errno=0
func=exp2 op1=408fffff.ffffffff result=7fefffff.fffffd3a.37a errno=0
func=exp2 op1=40900000.00000000 result=7ff00000.00000000 errno=ERANGE status=ox
func=exp2 op1=c090ca00.00000000 result=00000000.00000000.b50 status=ux
func=exp2 op1=c090cc00.00000000 result=00000000.00000000 errno=ERANGE status=ux
