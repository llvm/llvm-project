; Directed test cases for pow
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func=pow op1=00000000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000000 op2=00000000.00000001 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=00100000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=1fffffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=3bdfffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=3be00000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=3fe00000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=3ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=40000000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=40080000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=40120000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=40180000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=407ff800.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=408ff800.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=43dfffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=43e00000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=7fefffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=00000000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=00000000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000000 op2=80000000.00000001 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=80100000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=9fffffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=bbdfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=bbe00000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=bfe00000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=bff00000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=c0000000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=c0080000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=c0120000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=c0180000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=c07f3000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=c090ce00.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=00000000.00000000 op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=00000000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=00000000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=00000000.00000001 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000001 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000001 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000001 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000001 op2=3bdfffff.ffffffff result=3fefffff.ffffffff.d17 errno=0
func=pow op1=00000000.00000001 op2=3be00000.00000000 result=3fefffff.ffffffff.d17 errno=0
func=pow op1=00000000.00000001 op2=3fe00000.00000000 result=1e600000.00000000 errno=0
func=pow op1=00000000.00000001 op2=3ff00000.00000000 result=00000000.00000001 errno=0 status=u
func=pow op1=00000000.00000001 op2=40000000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00000000.00000001 op2=40080000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00000000.00000001 op2=40120000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00000000.00000001 op2=40180000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00000000.00000001 op2=407ff800.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00000000.00000001 op2=408ff800.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00000000.00000001 op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00000000.00000001 op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00000000.00000001 op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00000000.00000001 op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=00000000.00000001 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=00000000.00000001 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=00000000.00000001 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000001 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000001 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000001 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=00000000.00000001 op2=bbdfffff.ffffffff result=3ff00000.00000000.174 errno=0
func=pow op1=00000000.00000001 op2=bbe00000.00000000 result=3ff00000.00000000.174 errno=0
func=pow op1=00000000.00000001 op2=bfe00000.00000000 result=61800000.00000000 errno=0
func=pow op1=00000000.00000001 op2=bff00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=c0000000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=c0080000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=c0120000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=c0180000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=c07f3000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=c090ce00.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00000000.00000001 op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=00000000.00000001 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=00000000.00000001 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=00100000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=3bdfffff.ffffffff result=3fefffff.ffffffff.d3b errno=0
func=pow op1=00100000.00000000 op2=3be00000.00000000 result=3fefffff.ffffffff.d3b errno=0
func=pow op1=00100000.00000000 op2=3fe00000.00000000 result=20000000.00000000 errno=0
func=pow op1=00100000.00000000 op2=3ff00000.00000000 result=00100000.00000000 errno=0
func=pow op1=00100000.00000000 op2=40000000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00100000.00000000 op2=40080000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00100000.00000000 op2=40120000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00100000.00000000 op2=40180000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00100000.00000000 op2=407ff800.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00100000.00000000 op2=408ff800.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00100000.00000000 op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00100000.00000000 op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00100000.00000000 op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=00100000.00000000 op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=00100000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=00100000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=00100000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=bbdfffff.ffffffff result=3ff00000.00000000.162 errno=0
func=pow op1=00100000.00000000 op2=bbe00000.00000000 result=3ff00000.00000000.162 errno=0
func=pow op1=00100000.00000000 op2=bfe00000.00000000 result=5fe00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=bff00000.00000000 result=7fd00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=c0000000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00100000.00000000 op2=c0080000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00100000.00000000 op2=c0120000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00100000.00000000 op2=c0180000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00100000.00000000 op2=c07f3000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00100000.00000000 op2=c090ce00.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00100000.00000000 op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00100000.00000000 op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00100000.00000000 op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=00100000.00000000 op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=00100000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=00100000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=3bdfffff.ffffffff result=3fefffff.ffffffff.e9d errno=0
func=pow op1=1fffffff.ffffffff op2=3be00000.00000000 result=3fefffff.ffffffff.e9d errno=0
func=pow op1=1fffffff.ffffffff op2=3fe00000.00000000 result=2ff6a09e.667f3bcc.360 errno=0
func=pow op1=1fffffff.ffffffff op2=3ff00000.00000000 result=1fffffff.ffffffff errno=0
func=pow op1=1fffffff.ffffffff op2=40000000.00000000 result=000fffff.ffffffff errno=0 status=u
func=pow op1=1fffffff.ffffffff op2=40080000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=1fffffff.ffffffff op2=40120000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=1fffffff.ffffffff op2=40180000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=1fffffff.ffffffff op2=407ff800.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=1fffffff.ffffffff op2=408ff800.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=1fffffff.ffffffff op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=1fffffff.ffffffff op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=1fffffff.ffffffff op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=1fffffff.ffffffff op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=1fffffff.ffffffff op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=bbdfffff.ffffffff result=3ff00000.00000000.0b1 errno=0
func=pow op1=1fffffff.ffffffff op2=bbe00000.00000000 result=3ff00000.00000000.0b1 errno=0
func=pow op1=1fffffff.ffffffff op2=bfe00000.00000000 result=4fe6a09e.667f3bcc.eb0 errno=0
func=pow op1=1fffffff.ffffffff op2=bff00000.00000000 result=5fe00000.00000000.800 errno=0
func=pow op1=1fffffff.ffffffff op2=c0000000.00000000 result=7fd00000.00000001 errno=0
func=pow op1=1fffffff.ffffffff op2=c0080000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=1fffffff.ffffffff op2=c0120000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=1fffffff.ffffffff op2=c0180000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=1fffffff.ffffffff op2=c07f3000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=1fffffff.ffffffff op2=c090ce00.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=1fffffff.ffffffff op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=1fffffff.ffffffff op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=1fffffff.ffffffff op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=1fffffff.ffffffff op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=1fffffff.ffffffff op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=1fffffff.ffffffff op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=3bdfffff.ffffffff result=3fefffff.ffffffff.fff errno=0
func=pow op1=3fe00000.00000000 op2=3be00000.00000000 result=3fefffff.ffffffff.fff errno=0
func=pow op1=3fe00000.00000000 op2=3fe00000.00000000 result=3fe6a09e.667f3bcc.908 errno=0
func=pow op1=3fe00000.00000000 op2=3ff00000.00000000 result=3fe00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=40000000.00000000 result=3fd00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=40080000.00000000 result=3fc00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=40120000.00000000 result=3fa6a09e.667f3bcc.908 errno=0
func=pow op1=3fe00000.00000000 op2=40180000.00000000 result=3f900000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=407ff800.00000000 result=1ff6a09e.667f3bcc.908 errno=0
func=pow op1=3fe00000.00000000 op2=408ff800.00000000 result=00080000.00000000 errno=0 status=u
func=pow op1=3fe00000.00000000 op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3fe00000.00000000 op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3fe00000.00000000 op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3fe00000.00000000 op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3fe00000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=bbdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=bbe00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=bfe00000.00000000 result=3ff6a09e.667f3bcc.908 errno=0
func=pow op1=3fe00000.00000000 op2=bff00000.00000000 result=40000000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=c0000000.00000000 result=40100000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=c0080000.00000000 result=40200000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=c0120000.00000000 result=4036a09e.667f3bcc.908 errno=0
func=pow op1=3fe00000.00000000 op2=c0180000.00000000 result=40500000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=c07f3000.00000000 result=5f200000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=c090ce00.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fe00000.00000000 op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fe00000.00000000 op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fe00000.00000000 op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fe00000.00000000 op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=3fe00000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3fe00000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=3bdfffff.ffffffff result=3fefffff.ffffffff.fff errno=0
func=pow op1=3fef9800.00000000 op2=3be00000.00000000 result=3fefffff.ffffffff.fff errno=0
func=pow op1=3fef9800.00000000 op2=3fe00000.00000000 result=3fefcbd5.7acb4a6e.860 errno=0
func=pow op1=3fef9800.00000000 op2=3ff00000.00000000 result=3fef9800.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=40000000.00000000 result=3fef3152.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=40080000.00000000 result=3feecbf1.b5800000 errno=0
func=pow op1=3fef9800.00000000 op2=40120000.00000000 result=3fee3649.b95eb051.74b errno=0
func=pow op1=3fef9800.00000000 op2=40180000.00000000 result=3feda378.fe2081dd.720 errno=0
func=pow op1=3fef9800.00000000 op2=407ff800.00000000 result=3f57c7a0.fdc7f7ec.294 errno=0
func=pow op1=3fef9800.00000000 op2=408ff800.00000000 result=3ec1abd4.ca4dcd2b.5aa errno=0
func=pow op1=3fef9800.00000000 op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3fef9800.00000000 op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3fef9800.00000000 op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3fef9800.00000000 op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3fef9800.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=bbdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=bbe00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=bfe00000.00000000 result=3ff01a40.0d91bee3.a6e errno=0
func=pow op1=3fef9800.00000000 op2=bff00000.00000000 result=3ff034ab.2c50040d.2ac errno=0
func=pow op1=3fef9800.00000000 op2=c0000000.00000000 result=3ff06a03.b86753dd.bb6 errno=0
func=pow op1=3fef9800.00000000 op2=c0080000.00000000 result=3ff0a00b.defc06f4.558 errno=0
func=pow op1=3fef9800.00000000 op2=c0120000.00000000 result=3ff0f266.4d09b66a.72f errno=0
func=pow op1=3fef9800.00000000 op2=c0180000.00000000 result=3ff14658.ab6c8d31.ec8 errno=0
func=pow op1=3fef9800.00000000 op2=c07f3000.00000000 result=40825a4f.79ba0328.8d7 errno=0
func=pow op1=3fef9800.00000000 op2=c090ce00.00000000 result=412c5521.b1a8d47f.54d errno=0
func=pow op1=3fef9800.00000000 op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fef9800.00000000 op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fef9800.00000000 op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fef9800.00000000 op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=3fef9800.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3fef9800.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3fefffff.ffffffe0 op2=4386128b.68cf9fbc result=003ff70a.f0af9c79.372 errno=0
func=pow op1=3fefffff.ffffffe0 op2=c386128b.68cf9fbc result=7fa0047b.c8f04d90.332 errno=0
func=pow op1=3fefffff.ffffffff op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=3bdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=3be00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=3fe00000.00000000 result=3fefffff.ffffffff.800 errno=0
func=pow op1=3fefffff.ffffffff op2=3ff00000.00000000 result=3fefffff.ffffffff errno=0
func=pow op1=3fefffff.ffffffff op2=40000000.00000000 result=3fefffff.fffffffe errno=0
func=pow op1=3fefffff.ffffffff op2=40080000.00000000 result=3fefffff.fffffffd errno=0
func=pow op1=3fefffff.ffffffff op2=40120000.00000000 result=3fefffff.fffffffb.800 errno=0
func=pow op1=3fefffff.ffffffff op2=40180000.00000000 result=3fefffff.fffffffa errno=0
func=pow op1=3fefffff.ffffffff op2=407ff800.00000000 result=3fefffff.fffffe00.800 errno=0
func=pow op1=3fefffff.ffffffff op2=408ff800.00000000 result=3fefffff.fffffc01 errno=0
func=pow op1=3fefffff.ffffffff op2=4386128b.68cf9fbc result=3df1d45f.3e91e17c.d0c errno=0
func=pow op1=3fefffff.ffffffff op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3fefffff.ffffffff op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3fefffff.ffffffff op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3fefffff.ffffffff op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3fefffff.ffffffff op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=bbdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=bbe00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=bfe00000.00000000 result=3ff00000.00000000.400 errno=0
func=pow op1=3fefffff.ffffffff op2=bff00000.00000000 result=3ff00000.00000000.800 errno=0
func=pow op1=3fefffff.ffffffff op2=c0000000.00000000 result=3ff00000.00000001 errno=0
func=pow op1=3fefffff.ffffffff op2=c0080000.00000000 result=3ff00000.00000001.800 errno=0
func=pow op1=3fefffff.ffffffff op2=c0120000.00000000 result=3ff00000.00000002.400 errno=0
func=pow op1=3fefffff.ffffffff op2=c0180000.00000000 result=3ff00000.00000003 errno=0
func=pow op1=3fefffff.ffffffff op2=c07f3000.00000000 result=3ff00000.000000f9.800 errno=0
func=pow op1=3fefffff.ffffffff op2=c090ce00.00000000 result=3ff00000.00000219.c00 errno=0
func=pow op1=3fefffff.ffffffff op2=c386128b.68cf9fbc result=41ecb761.33b97fcc.60b errno=0
func=pow op1=3fefffff.ffffffff op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fefffff.ffffffff op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fefffff.ffffffff op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3fefffff.ffffffff op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=3fefffff.ffffffff op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3fefffff.ffffffff op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=3bdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=3be00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=3fe00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=3ff00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=40000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=40080000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=40120000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=40180000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=407ff800.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=408ff800.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=43dfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=43e00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=7fefffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=7ff00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3ff00000.00000000 op2=7ff80000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=bbdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=bbe00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=bfe00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=bff00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=c0000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=c0080000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=c0120000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=c0180000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=c07f3000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=c090ce00.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=c3dfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=c3e00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=ffefffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=fff00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3ff00000.00000000 op2=fff80000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=3bdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=3be00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=3fe00000.00000000 result=3ff00000.00000000.800 errno=0
func=pow op1=3ff00000.00000001 op2=3ff00000.00000000 result=3ff00000.00000001 errno=0
func=pow op1=3ff00000.00000001 op2=40000000.00000000 result=3ff00000.00000002 errno=0
func=pow op1=3ff00000.00000001 op2=40080000.00000000 result=3ff00000.00000003 errno=0
func=pow op1=3ff00000.00000001 op2=40120000.00000000 result=3ff00000.00000004.800 errno=0
func=pow op1=3ff00000.00000001 op2=40180000.00000000 result=3ff00000.00000006 errno=0
func=pow op1=3ff00000.00000001 op2=407ff800.00000000 result=3ff00000.000001ff.800 errno=0
func=pow op1=3ff00000.00000001 op2=408ff800.00000000 result=3ff00000.000003ff errno=0
func=pow op1=3ff00000.00000001 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3ff00000.00000001 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3ff00000.00000001 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3ff00000.00000001 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3ff00000.00000001 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=bbdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=bbe00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=bfe00000.00000000 result=3fefffff.ffffffff errno=0
func=pow op1=3ff00000.00000001 op2=bff00000.00000000 result=3fefffff.fffffffe errno=0
func=pow op1=3ff00000.00000001 op2=c0000000.00000000 result=3fefffff.fffffffc errno=0
func=pow op1=3ff00000.00000001 op2=c0080000.00000000 result=3fefffff.fffffffa errno=0
func=pow op1=3ff00000.00000001 op2=c0120000.00000000 result=3fefffff.fffffff7 errno=0
func=pow op1=3ff00000.00000001 op2=c0180000.00000000 result=3fefffff.fffffff4 errno=0
func=pow op1=3ff00000.00000001 op2=c07f3000.00000000 result=3fefffff.fffffc1a errno=0
func=pow op1=3ff00000.00000001 op2=c090ce00.00000000 result=3fefffff.fffff799 errno=0
func=pow op1=3ff00000.00000001 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3ff00000.00000001 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3ff00000.00000001 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3ff00000.00000001 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=3ff00000.00000001 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3ff00000.00000001 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=3bdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=3be00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=3fe00000.00000000 result=3ff019eb.020ee283.520 errno=0
func=pow op1=3ff03400.00000000 op2=3ff00000.00000000 result=3ff03400.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=40000000.00000000 result=3ff068a9.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=40080000.00000000 result=3ff09dfd.25400000 errno=0
func=pow op1=3ff03400.00000000 op2=40120000.00000000 result=3ff0ef41.05a27f91.ece errno=0
func=pow op1=3ff03400.00000000 op2=40180000.00000000 result=3ff14212.5220325e.b90 errno=0
func=pow op1=3ff03400.00000000 op2=407ff800.00000000 result=4083d3b3.8a3213c3.297 errno=0
func=pow op1=3ff03400.00000000 op2=408ff800.00000000 result=411891bb.7f728082.d88 errno=0
func=pow op1=3ff03400.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3ff03400.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3ff03400.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=3ff03400.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3ff03400.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=bbdfffff.ffffffff result=3fefffff.ffffffff.fff errno=0
func=pow op1=3ff03400.00000000 op2=bbe00000.00000000 result=3fefffff.ffffffff.fff errno=0
func=pow op1=3ff03400.00000000 op2=bfe00000.00000000 result=3fefcc7d.6c7d2e30.865 errno=0
func=pow op1=3ff03400.00000000 op2=bff00000.00000000 result=3fef994d.c3455e8c.b6a errno=0
func=pow op1=3ff03400.00000000 op2=c0000000.00000000 result=3fef33e5.1aaea6ee.309 errno=0
func=pow op1=3ff03400.00000000 op2=c0080000.00000000 result=3feecfc1.e487ed2b.638 errno=0
func=pow op1=3ff03400.00000000 op2=c0120000.00000000 result=3fee3be6.60bd4449.151 errno=0
func=pow op1=3ff03400.00000000 op2=c0180000.00000000 result=3fedaad0.65924e45.6c1 errno=0
func=pow op1=3ff03400.00000000 op2=c07f3000.00000000 result=3f5e3bf6.471a7841.69b errno=0
func=pow op1=3ff03400.00000000 op2=c090ce00.00000000 result=3eb57de0.09c1a44f.f1a errno=0
func=pow op1=3ff03400.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3ff03400.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3ff03400.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=3ff03400.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=3ff03400.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=3ff03400.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=40000000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=3bdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=3be00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=3fe00000.00000000 result=3ff6a09e.667f3bcc.908 errno=0
func=pow op1=40000000.00000000 op2=3ff00000.00000000 result=40000000.00000000 errno=0
func=pow op1=40000000.00000000 op2=40000000.00000000 result=40100000.00000000 errno=0
func=pow op1=40000000.00000000 op2=40080000.00000000 result=40200000.00000000 errno=0
func=pow op1=40000000.00000000 op2=40120000.00000000 result=4036a09e.667f3bcc.908 errno=0
func=pow op1=40000000.00000000 op2=40180000.00000000 result=40500000.00000000 errno=0
func=pow op1=40000000.00000000 op2=407ff800.00000000 result=5fe6a09e.667f3bcc.908 errno=0
func=pow op1=40000000.00000000 op2=408ff800.00000000 result=7fe00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40000000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40000000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40000000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=40000000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=40000000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=bbdfffff.ffffffff result=3fefffff.ffffffff.fff errno=0
func=pow op1=40000000.00000000 op2=bbe00000.00000000 result=3fefffff.ffffffff.fff errno=0
func=pow op1=40000000.00000000 op2=bfe00000.00000000 result=3fe6a09e.667f3bcc.908 errno=0
func=pow op1=40000000.00000000 op2=bff00000.00000000 result=3fe00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=c0000000.00000000 result=3fd00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=c0080000.00000000 result=3fc00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=c0120000.00000000 result=3fa6a09e.667f3bcc.908 errno=0
func=pow op1=40000000.00000000 op2=c0180000.00000000 result=3f900000.00000000 errno=0
func=pow op1=40000000.00000000 op2=c07f3000.00000000 result=20c00000.00000000 errno=0
func=pow op1=40000000.00000000 op2=c08f3a00.00000000 result=017ae89f.995ad3ad.5e8 errno=0
func=pow op1=40000000.00000000 op2=c090ce00.00000000 result=00000000.00000000.5a8 errno=ERANGE status=u
func=pow op1=40000000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40000000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40000000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40000000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=40000000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=40000000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=40080000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=3bdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=3be00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=3fe00000.00000000 result=3ffbb67a.e8584caa.73b errno=0
func=pow op1=40080000.00000000 op2=3ff00000.00000000 result=40080000.00000000 errno=0
func=pow op1=40080000.00000000 op2=40000000.00000000 result=40220000.00000000 errno=0
func=pow op1=40080000.00000000 op2=40080000.00000000 result=403b0000.00000000 errno=0
func=pow op1=40080000.00000000 op2=40120000.00000000 result=40618979.c707e083.dd3 errno=0
func=pow op1=40080000.00000000 op2=40180000.00000000 result=4086c800.00000000 errno=0
func=pow op1=40080000.00000000 op2=407ff800.00000000 result=729a2473.a65e6847.3ca errno=0
func=pow op1=40080000.00000000 op2=408ff800.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40080000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40080000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40080000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40080000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=40080000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=40080000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40080000.00000000 op2=bbdfffff.ffffffff result=3fefffff.ffffffff.ffe errno=0
func=pow op1=40080000.00000000 op2=bbe00000.00000000 result=3fefffff.ffffffff.ffe errno=0
func=pow op1=40080000.00000000 op2=bfe00000.00000000 result=3fe279a7.4590331c.4d2 errno=0
func=pow op1=40080000.00000000 op2=bff00000.00000000 result=3fd55555.55555555.555 errno=0
func=pow op1=40080000.00000000 op2=c0000000.00000000 result=3fbc71c7.1c71c71c.71c errno=0
func=pow op1=40080000.00000000 op2=c0080000.00000000 result=3fa2f684.bda12f68.4bd errno=0
func=pow op1=40080000.00000000 op2=c0120000.00000000 result=3f7d3205.2b8e89a7.fb7 errno=0
func=pow op1=40080000.00000000 op2=c0180000.00000000 result=3f567980.e0bf08c7.765 errno=0
func=pow op1=40080000.00000000 op2=c07f3000.00000000 result=0e81314b.59b2f0d0.9a8 errno=0
func=pow op1=40080000.00000000 op2=c090ce00.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40080000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40080000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40080000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40080000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=40080000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=40080000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=40120000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=3bdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=3be00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=3fe00000.00000000 result=4000f876.ccdf6cd9.6c6 errno=0
func=pow op1=40120000.00000000 op2=3ff00000.00000000 result=40120000.00000000 errno=0
func=pow op1=40120000.00000000 op2=40000000.00000000 result=40344000.00000000 errno=0
func=pow op1=40120000.00000000 op2=40080000.00000000 result=4056c800.00000000 errno=0
func=pow op1=40120000.00000000 op2=40120000.00000000 result=408b2efd.cb8aa24b.053 errno=0
func=pow op1=40120000.00000000 op2=40180000.00000000 result=40c037e2.00000000 errno=0
func=pow op1=40120000.00000000 op2=407ff800.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40120000.00000000 op2=408ff800.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40120000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40120000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40120000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40120000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=40120000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=40120000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40120000.00000000 op2=bbdfffff.ffffffff result=3fefffff.ffffffff.ffe errno=0
func=pow op1=40120000.00000000 op2=bbe00000.00000000 result=3fefffff.ffffffff.ffe errno=0
func=pow op1=40120000.00000000 op2=bfe00000.00000000 result=3fde2b7d.ddfefa66.160 errno=0
func=pow op1=40120000.00000000 op2=bff00000.00000000 result=3fcc71c7.1c71c71c.71c errno=0
func=pow op1=40120000.00000000 op2=c0000000.00000000 result=3fa948b0.fcd6e9e0.652 errno=0
func=pow op1=40120000.00000000 op2=c0080000.00000000 result=3f867980.e0bf08c7.765 errno=0
func=pow op1=40120000.00000000 op2=c0120000.00000000 result=3f52d5bc.e225fd84.857 errno=0
func=pow op1=40120000.00000000 op2=c0180000.00000000 result=3f1f91bd.1b62b9ce.c8a errno=0
func=pow op1=40120000.00000000 op2=c07f3000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40120000.00000000 op2=c090ce00.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40120000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40120000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40120000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40120000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=40120000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=40120000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=40180000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=3bdfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=3be00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=3fe00000.00000000 result=4003988e.1409212e.7d0 errno=0
func=pow op1=40180000.00000000 op2=3ff00000.00000000 result=40180000.00000000 errno=0
func=pow op1=40180000.00000000 op2=40000000.00000000 result=40420000.00000000 errno=0
func=pow op1=40180000.00000000 op2=40080000.00000000 result=406b0000.00000000 errno=0
func=pow op1=40180000.00000000 op2=40120000.00000000 result=40a8cd13.d15b8dfe.d63 errno=0
func=pow op1=40180000.00000000 op2=40180000.00000000 result=40e6c800.00000000 errno=0
func=pow op1=40180000.00000000 op2=407ff800.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40180000.00000000 op2=408ff800.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40180000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40180000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40180000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=40180000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=40180000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=40180000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=40180000.00000000 op2=bbdfffff.ffffffff result=3fefffff.ffffffff.ffe errno=0
func=pow op1=40180000.00000000 op2=bbe00000.00000000 result=3fefffff.ffffffff.ffe errno=0
func=pow op1=40180000.00000000 op2=bfe00000.00000000 result=3fda20bd.700c2c3d.fc0 errno=0
func=pow op1=40180000.00000000 op2=bff00000.00000000 result=3fc55555.55555555.555 errno=0
func=pow op1=40180000.00000000 op2=c0000000.00000000 result=3f9c71c7.1c71c71c.71c errno=0
func=pow op1=40180000.00000000 op2=c0080000.00000000 result=3f72f684.bda12f68.4bd errno=0
func=pow op1=40180000.00000000 op2=c0120000.00000000 result=3f34a4ee.2c48d3f1.c3f errno=0
func=pow op1=40180000.00000000 op2=c0180000.00000000 result=3ef67980.e0bf08c7.765 errno=0
func=pow op1=40180000.00000000 op2=c07f3000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40180000.00000000 op2=c090ce00.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40180000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40180000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40180000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=40180000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=40180000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=40180000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=3bdfffff.ffffffff result=3ff00000.00000000.053 errno=0
func=pow op1=4f0fffff.ffffffff op2=3be00000.00000000 result=3ff00000.00000000.053 errno=0
func=pow op1=4f0fffff.ffffffff op2=3fe00000.00000000 result=477fffff.ffffffff.800 errno=0
func=pow op1=4f0fffff.ffffffff op2=3ff00000.00000000 result=4f0fffff.ffffffff errno=0
func=pow op1=4f0fffff.ffffffff op2=40000000.00000000 result=5e2fffff.fffffffe errno=0
func=pow op1=4f0fffff.ffffffff op2=40080000.00000000 result=6d4fffff.fffffffd errno=0
func=pow op1=4f0fffff.ffffffff op2=40120000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=4f0fffff.ffffffff op2=40180000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=4f0fffff.ffffffff op2=407ff800.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=4f0fffff.ffffffff op2=408ff800.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=4f0fffff.ffffffff op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=4f0fffff.ffffffff op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=4f0fffff.ffffffff op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=4f0fffff.ffffffff op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=4f0fffff.ffffffff op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=bbdfffff.ffffffff result=3fefffff.ffffffff.f58 errno=0
func=pow op1=4f0fffff.ffffffff op2=bbe00000.00000000 result=3fefffff.ffffffff.f58 errno=0
func=pow op1=4f0fffff.ffffffff op2=bfe00000.00000000 result=38600000.00000000.400 errno=0
func=pow op1=4f0fffff.ffffffff op2=bff00000.00000000 result=30d00000.00000000.800 errno=0
func=pow op1=4f0fffff.ffffffff op2=c0000000.00000000 result=21b00000.00000001 errno=0
func=pow op1=4f0fffff.ffffffff op2=c0080000.00000000 result=12900000.00000001.800 errno=0
func=pow op1=4f0fffff.ffffffff op2=c0120000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=4f0fffff.ffffffff op2=c0180000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=4f0fffff.ffffffff op2=c07f3000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=4f0fffff.ffffffff op2=c090ce00.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=4f0fffff.ffffffff op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=4f0fffff.ffffffff op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=4f0fffff.ffffffff op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=4f0fffff.ffffffff op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=4f0fffff.ffffffff op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=4f0fffff.ffffffff op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=00000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=00100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=1fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=3bdfffff.ffffffff result=3ff00000.00000000.162 errno=0
func=pow op1=7fefffff.ffffffff op2=3be00000.00000000 result=3ff00000.00000000.162 errno=0
func=pow op1=7fefffff.ffffffff op2=3fe00000.00000000 result=5fefffff.ffffffff.800 errno=0
func=pow op1=7fefffff.ffffffff op2=3ff00000.00000000 result=7fefffff.ffffffff errno=0
func=pow op1=7fefffff.ffffffff op2=40000000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=40080000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=40120000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=40180000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=407ff800.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=408ff800.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=4386128b.68cf9fbc result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=7fefffff.ffffffff op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7fefffff.ffffffff op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=80000000.00000001 result=3ff00000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=80100000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=9fffffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=bbdfffff.ffffffff result=3fefffff.ffffffff.d3a errno=0
func=pow op1=7fefffff.ffffffff op2=bbe00000.00000000 result=3fefffff.ffffffff.d3a errno=0
func=pow op1=7fefffff.ffffffff op2=bfe00000.00000000 result=1ff00000.00000000.400 errno=0
func=pow op1=7fefffff.ffffffff op2=bff00000.00000000 result=00040000.00000000.200 errno=0 status=u
func=pow op1=7fefffff.ffffffff op2=c0000000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=c0080000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=c0120000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=c0180000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=c07f3000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=c090ce00.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=c386128b.68cf9fbc result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=7fefffff.ffffffff op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=7fefffff.ffffffff op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7fefffff.ffffffff op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=00000000.00000001 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=00100000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=1fffffff.ffffffff result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=3bdfffff.ffffffff result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=3be00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=3fe00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=3ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=40000000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=40080000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=40120000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=40180000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=407ff800.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=408ff800.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=80000000.00000001 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=80100000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=9fffffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=bbdfffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=bbe00000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=bfe00000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=bff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=c0000000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=c0080000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=c0120000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=c0180000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=c07f3000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=c090ce00.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=7ff00000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=7ff00000.00000001 op2=00000000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=00000000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=00100000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=3be00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=3fe00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=3ff00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=40000000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=40080000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=40120000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=40180000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=407ff800.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=408ff800.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=43dfffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=43e00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=7fefffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=7ff00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=80000000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=80000000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=80100000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=bbe00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=bfe00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=bff00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=c0000000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=c0080000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=c0120000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=c0180000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=c07f3000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=c090ce00.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=c3dfffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=c3e00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=ffefffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=fff00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff00000.00000001 op2=fff80000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff80000.00000001 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=00000000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=00100000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=3be00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=3fe00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=3ff00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=40000000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=40080000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=40120000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=40180000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=407ff800.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=408ff800.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=43dfffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=43e00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=7fefffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=7ff00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff80000.00000001 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=80000000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=80100000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=bbe00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=bfe00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=bff00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=c0000000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=c0080000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=c0120000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=c0180000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=c07f3000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=c090ce00.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=c3dfffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=c3e00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=ffefffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=fff00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=7ff80000.00000001 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=7ff80000.00000001 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=80000000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=80000000.00000000 op2=00000000.00000001 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=00100000.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=1fffffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=3bdfffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=3be00000.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=3fe00000.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=3ff00000.00000000 result=80000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=40000000.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=40080000.00000000 result=80000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=40120000.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=40180000.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=407ff800.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=408ff800.00000000 result=80000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=43dfffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=43e00000.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=7fefffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=80000000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=80000000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=80000000.00000000 op2=80000000.00000001 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=80100000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=9fffffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=bbdfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=bbe00000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=bfe00000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=bff00000.00000000 result=fff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=c0000000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=c0080000.00000000 result=fff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=c0120000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=c0180000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=c07f3000.00000000 result=fff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=c090ce00.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=z
func=pow op1=80000000.00000000 op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=80000000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=80000000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=80000000.00000001 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=80000000.00000001 op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=3ff00000.00000000 result=80000000.00000001 errno=0 status=u
func=pow op1=80000000.00000001 op2=40000000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80000000.00000001 op2=40080000.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=80000000.00000001 op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=40180000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80000000.00000001 op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=408ff800.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=80000000.00000001 op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80000000.00000001 op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80000000.00000001 op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80000000.00000001 op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=80000000.00000001 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=80000000.00000001 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=80000000.00000001 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=80000000.00000001 op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=bff00000.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=80000000.00000001 op2=c0000000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80000000.00000001 op2=c0080000.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=80000000.00000001 op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=c0180000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80000000.00000001 op2=c07f3000.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=80000000.00000001 op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80000000.00000001 op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80000000.00000001 op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80000000.00000001 op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80000000.00000001 op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=80000000.00000001 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=80000000.00000001 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=80100000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=80100000.00000000 op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=3ff00000.00000000 result=80100000.00000000 errno=0
func=pow op1=80100000.00000000 op2=40000000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80100000.00000000 op2=40080000.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=80100000.00000000 op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=40180000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80100000.00000000 op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=408ff800.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=80100000.00000000 op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80100000.00000000 op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80100000.00000000 op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=80100000.00000000 op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=80100000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=80100000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=80100000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=80100000.00000000 op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=bff00000.00000000 result=ffd00000.00000000 errno=0
func=pow op1=80100000.00000000 op2=c0000000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80100000.00000000 op2=c0080000.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=80100000.00000000 op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=c0180000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80100000.00000000 op2=c07f3000.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=80100000.00000000 op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=80100000.00000000 op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80100000.00000000 op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80100000.00000000 op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=80100000.00000000 op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=80100000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=80100000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=9fffffff.ffffffff op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=9fffffff.ffffffff op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=3ff00000.00000000 result=9fffffff.ffffffff errno=0
func=pow op1=9fffffff.ffffffff op2=40000000.00000000 result=000fffff.ffffffff errno=0 status=u
func=pow op1=9fffffff.ffffffff op2=40080000.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=9fffffff.ffffffff op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=40180000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=9fffffff.ffffffff op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=408ff800.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=9fffffff.ffffffff op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=9fffffff.ffffffff op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=9fffffff.ffffffff op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=9fffffff.ffffffff op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=9fffffff.ffffffff op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=9fffffff.ffffffff op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=9fffffff.ffffffff op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=9fffffff.ffffffff op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=bff00000.00000000 result=dfe00000.00000000.800 errno=0
func=pow op1=9fffffff.ffffffff op2=c0000000.00000000 result=7fd00000.00000001 errno=0
func=pow op1=9fffffff.ffffffff op2=c0080000.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=9fffffff.ffffffff op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=c0180000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=9fffffff.ffffffff op2=c07f3000.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=9fffffff.ffffffff op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=9fffffff.ffffffff op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=9fffffff.ffffffff op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=9fffffff.ffffffff op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=9fffffff.ffffffff op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=9fffffff.ffffffff op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=9fffffff.ffffffff op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=3ff00000.00000000 result=bfe00000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=40000000.00000000 result=3fd00000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=40080000.00000000 result=bfc00000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=40180000.00000000 result=3f900000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=408ff800.00000000 result=80080000.00000000 errno=0 status=u
func=pow op1=bfe00000.00000000 op2=4386128b.68cf9fbc result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bfe00000.00000000 op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bfe00000.00000000 op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bfe00000.00000000 op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bfe00000.00000000 op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=bfe00000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=bff00000.00000000 result=c0000000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=c0000000.00000000 result=40100000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=c0080000.00000000 result=c0200000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=c0180000.00000000 result=40500000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=c07f3000.00000000 result=df200000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfe00000.00000000 op2=c386128b.68cf9fbc result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bfe00000.00000000 op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bfe00000.00000000 op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bfe00000.00000000 op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bfe00000.00000000 op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=bfe00000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=bfe00000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=bfefffff.ffffffe0 op2=4386128b.68cf9fbc result=003ff70a.f0af9c79.372 errno=0
func=pow op1=bfefffff.ffffffe0 op2=c386128b.68cf9fbc result=7fa0047b.c8f04d90.332 errno=0
func=pow op1=bfefffff.ffffffff op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bfefffff.ffffffff op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=3ff00000.00000000 result=bfefffff.ffffffff errno=0
func=pow op1=bfefffff.ffffffff op2=40000000.00000000 result=3fefffff.fffffffe errno=0
func=pow op1=bfefffff.ffffffff op2=40080000.00000000 result=bfefffff.fffffffd errno=0
func=pow op1=bfefffff.ffffffff op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=40180000.00000000 result=3fefffff.fffffffa errno=0
func=pow op1=bfefffff.ffffffff op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=408ff800.00000000 result=bfefffff.fffffc01 errno=0
func=pow op1=bfefffff.ffffffff op2=4386128b.68cf9fbc result=3df1d45f.3e91e17c.d0c errno=0
func=pow op1=bfefffff.ffffffff op2=43dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bfefffff.ffffffff op2=43e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bfefffff.ffffffff op2=7fefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bfefffff.ffffffff op2=7ff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=bfefffff.ffffffff op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=bfefffff.ffffffff op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=bfefffff.ffffffff op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bfefffff.ffffffff op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=bff00000.00000000 result=bff00000.00000000.800 errno=0
func=pow op1=bfefffff.ffffffff op2=c0000000.00000000 result=3ff00000.00000001 errno=0
func=pow op1=bfefffff.ffffffff op2=c0080000.00000000 result=bff00000.00000001.800 errno=0
func=pow op1=bfefffff.ffffffff op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=c0180000.00000000 result=3ff00000.00000003 errno=0
func=pow op1=bfefffff.ffffffff op2=c07f3000.00000000 result=bff00000.000000f9.800 errno=0
func=pow op1=bfefffff.ffffffff op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bfefffff.ffffffff op2=c386128b.68cf9fbc result=41ecb761.33b97fcc.60b errno=0
func=pow op1=bfefffff.ffffffff op2=c3dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bfefffff.ffffffff op2=c3e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bfefffff.ffffffff op2=ffefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bfefffff.ffffffff op2=fff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=bfefffff.ffffffff op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=bfefffff.ffffffff op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=3ff00000.00000000 result=bff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=40000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=40080000.00000000 result=bff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=40180000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=408ff800.00000000 result=bff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=43dfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=43e00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=7fefffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=7ff00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=bff00000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=bff00000.00000000 result=bff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=c0000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=c0080000.00000000 result=bff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=c0180000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=c07f3000.00000000 result=bff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000000 op2=c3dfffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=c3e00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=ffefffff.ffffffff result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=fff00000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=bff00000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=bff00000.00000001 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000001 op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=3ff00000.00000000 result=bff00000.00000001 errno=0
func=pow op1=bff00000.00000001 op2=40000000.00000000 result=3ff00000.00000002 errno=0
func=pow op1=bff00000.00000001 op2=40080000.00000000 result=bff00000.00000003 errno=0
func=pow op1=bff00000.00000001 op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=40180000.00000000 result=3ff00000.00000006 errno=0
func=pow op1=bff00000.00000001 op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=408ff800.00000000 result=bff00000.000003ff errno=0
func=pow op1=bff00000.00000001 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bff00000.00000001 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bff00000.00000001 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=bff00000.00000001 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=bff00000.00000001 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=bff00000.00000001 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=bff00000.00000001 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=bff00000.00000001 op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=bff00000.00000000 result=bfefffff.fffffffe errno=0
func=pow op1=bff00000.00000001 op2=c0000000.00000000 result=3fefffff.fffffffc errno=0
func=pow op1=bff00000.00000001 op2=c0080000.00000000 result=bfefffff.fffffffa errno=0
func=pow op1=bff00000.00000001 op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=c0180000.00000000 result=3fefffff.fffffff4 errno=0
func=pow op1=bff00000.00000001 op2=c07f3000.00000000 result=bfefffff.fffffc1a errno=0
func=pow op1=bff00000.00000001 op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=bff00000.00000001 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bff00000.00000001 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bff00000.00000001 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=bff00000.00000001 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=bff00000.00000001 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=bff00000.00000001 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=3ff00000.00000000 result=c0000000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=40000000.00000000 result=40100000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=40080000.00000000 result=c0200000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=40180000.00000000 result=40500000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=408ff800.00000000 result=ffe00000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0000000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0000000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0000000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=c0000000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=bff00000.00000000 result=bfe00000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=c0000000.00000000 result=3fd00000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=c0080000.00000000 result=bfc00000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=c0180000.00000000 result=3f900000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=c07f3000.00000000 result=a0c00000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0000000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0000000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0000000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0000000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=c0000000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=c0000000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=c0080000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=c0080000.00000000 op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=3ff00000.00000000 result=c0080000.00000000 errno=0
func=pow op1=c0080000.00000000 op2=40000000.00000000 result=40220000.00000000 errno=0
func=pow op1=c0080000.00000000 op2=40080000.00000000 result=c03b0000.00000000 errno=0
func=pow op1=c0080000.00000000 op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=40180000.00000000 result=4086c800.00000000 errno=0
func=pow op1=c0080000.00000000 op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=408ff800.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=c0080000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0080000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0080000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0080000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=c0080000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=c0080000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=c0080000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=c0080000.00000000 op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=bff00000.00000000 result=bfd55555.55555555.555 errno=0
func=pow op1=c0080000.00000000 op2=c0000000.00000000 result=3fbc71c7.1c71c71c.71c errno=0
func=pow op1=c0080000.00000000 op2=c0080000.00000000 result=bfa2f684.bda12f68.4bd errno=0
func=pow op1=c0080000.00000000 op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=c0180000.00000000 result=3f567980.e0bf08c7.765 errno=0
func=pow op1=c0080000.00000000 op2=c07f3000.00000000 result=8e81314b.59b2f0d0.9a8 errno=0
func=pow op1=c0080000.00000000 op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0080000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0080000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0080000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0080000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=c0080000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=c0080000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=c0120000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=c0120000.00000000 op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=3ff00000.00000000 result=c0120000.00000000 errno=0
func=pow op1=c0120000.00000000 op2=40000000.00000000 result=40344000.00000000 errno=0
func=pow op1=c0120000.00000000 op2=40080000.00000000 result=c056c800.00000000 errno=0
func=pow op1=c0120000.00000000 op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=40180000.00000000 result=40c037e2.00000000 errno=0
func=pow op1=c0120000.00000000 op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=408ff800.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=c0120000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0120000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0120000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0120000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=c0120000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=c0120000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=c0120000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=c0120000.00000000 op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=bff00000.00000000 result=bfcc71c7.1c71c71c.71c errno=0
func=pow op1=c0120000.00000000 op2=c0000000.00000000 result=3fa948b0.fcd6e9e0.652 errno=0
func=pow op1=c0120000.00000000 op2=c0080000.00000000 result=bf867980.e0bf08c7.765 errno=0
func=pow op1=c0120000.00000000 op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=c0180000.00000000 result=3f1f91bd.1b62b9ce.c8a errno=0
func=pow op1=c0120000.00000000 op2=c07f3000.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=c0120000.00000000 op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0120000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0120000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0120000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0120000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=c0120000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=c0120000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=c0180000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=c0180000.00000000 op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=3ff00000.00000000 result=c0180000.00000000 errno=0
func=pow op1=c0180000.00000000 op2=40000000.00000000 result=40420000.00000000 errno=0
func=pow op1=c0180000.00000000 op2=40080000.00000000 result=c06b0000.00000000 errno=0
func=pow op1=c0180000.00000000 op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=40180000.00000000 result=40e6c800.00000000 errno=0
func=pow op1=c0180000.00000000 op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=408ff800.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=c0180000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0180000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0180000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=c0180000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=c0180000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=c0180000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=c0180000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=c0180000.00000000 op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=bff00000.00000000 result=bfc55555.55555555.555 errno=0
func=pow op1=c0180000.00000000 op2=c0000000.00000000 result=3f9c71c7.1c71c71c.71c errno=0
func=pow op1=c0180000.00000000 op2=c0080000.00000000 result=bf72f684.bda12f68.4bd errno=0
func=pow op1=c0180000.00000000 op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=c0180000.00000000 result=3ef67980.e0bf08c7.765 errno=0
func=pow op1=c0180000.00000000 op2=c07f3000.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=c0180000.00000000 op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=c0180000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0180000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0180000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=c0180000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=c0180000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=c0180000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=ffefffff.ffffffff op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=ffefffff.ffffffff op2=00000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=00100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=1fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=3be00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=3fe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=3ff00000.00000000 result=ffefffff.ffffffff errno=0
func=pow op1=ffefffff.ffffffff op2=40000000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=ffefffff.ffffffff op2=40080000.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=ffefffff.ffffffff op2=40120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=40180000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=ffefffff.ffffffff op2=407ff800.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=408ff800.00000000 result=fff00000.00000000 errno=ERANGE status=o
func=pow op1=ffefffff.ffffffff op2=43dfffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=ffefffff.ffffffff op2=43e00000.00000000 result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=ffefffff.ffffffff op2=7fefffff.ffffffff result=7ff00000.00000000 errno=ERANGE status=o
func=pow op1=ffefffff.ffffffff op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=ffefffff.ffffffff op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=ffefffff.ffffffff op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=ffefffff.ffffffff op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=ffefffff.ffffffff op2=80000000.00000001 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=80100000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=9fffffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=bbe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=bfe00000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=bff00000.00000000 result=80040000.00000000.200 errno=0 status=u
func=pow op1=ffefffff.ffffffff op2=c0000000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=ffefffff.ffffffff op2=c0080000.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=ffefffff.ffffffff op2=c0120000.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=c0180000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=ffefffff.ffffffff op2=c07f3000.00000000 result=80000000.00000000 errno=ERANGE status=u
func=pow op1=ffefffff.ffffffff op2=c090ce00.00000000 result=7ff80000.00000000 errno=EDOM status=i
func=pow op1=ffefffff.ffffffff op2=c3dfffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=ffefffff.ffffffff op2=c3e00000.00000000 result=00000000.00000000 errno=ERANGE status=u
func=pow op1=ffefffff.ffffffff op2=ffefffff.ffffffff result=00000000.00000000 errno=ERANGE status=u
func=pow op1=ffefffff.ffffffff op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=ffefffff.ffffffff op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=ffefffff.ffffffff op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=00000000.00000001 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=00100000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=1fffffff.ffffffff result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=3bdfffff.ffffffff result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=3be00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=3fe00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=3ff00000.00000000 result=fff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=40000000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=40080000.00000000 result=fff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=40120000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=40180000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=407ff800.00000000 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=408ff800.00000000 result=fff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=43dfffff.ffffffff result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=43e00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=7fefffff.ffffffff result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=7ff00000.00000000 result=7ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000000 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=80000000.00000001 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=80100000.00000000 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=9fffffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=bbdfffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=bbe00000.00000000 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=bfe00000.00000000 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=bff00000.00000000 result=80000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=c0000000.00000000 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=c0080000.00000000 result=80000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=c0120000.00000000 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=c0180000.00000000 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=c07f3000.00000000 result=80000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=c090ce00.00000000 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=c3dfffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=c3e00000.00000000 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=ffefffff.ffffffff result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=fff00000.00000000 result=00000000.00000000 errno=0
func=pow op1=fff00000.00000000 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000000 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=fff00000.00000001 op2=00000000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=00000000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=00100000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=3be00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=3fe00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=3ff00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=40000000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=40080000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=40120000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=40180000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=407ff800.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=408ff800.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=43dfffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=43e00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=7fefffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=7ff00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=80000000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=80000000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=80100000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=bbe00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=bfe00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=bff00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=c0000000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=c0080000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=c0120000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=c0180000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=c07f3000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=c090ce00.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=c3dfffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=c3e00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=ffefffff.ffffffff result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=fff00000.00000000 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff00000.00000001 op2=fff80000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff80000.00000001 op2=00000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=00000000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=00100000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=1fffffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=3bdfffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=3be00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=3fe00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=3ff00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=40000000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=40080000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=40120000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=40180000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=407ff800.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=408ff800.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=43dfffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=43e00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=7fefffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=7ff00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=7ff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff80000.00000001 op2=7ff80000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=80000000.00000000 result=3ff00000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=80000000.00000001 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=80100000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=9fffffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=bbdfffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=bbe00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=bfe00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=bff00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=c0000000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=c0080000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=c0120000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=c0180000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=c07f3000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=c090ce00.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=c3dfffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=c3e00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=ffefffff.ffffffff result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=fff00000.00000000 result=7ff80000.00000000 errno=0
func=pow op1=fff80000.00000001 op2=fff00000.00000001 result=7ff80000.00000000 errno=0 status=i
func=pow op1=fff80000.00000001 op2=fff80000.00000001 result=7ff80000.00000000 errno=0
