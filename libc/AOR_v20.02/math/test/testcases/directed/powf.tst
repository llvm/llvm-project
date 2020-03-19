; powf.tst - Directed test cases for powf
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func=powf op1=7f800001 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=7fc00001 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=ffc00001 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=7f800000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=40800000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=40400000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=3f000000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=00000000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=80000000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=bf000000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=c0400000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=c0800000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=ff800000 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=7f800001 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=7fc00001 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=ffc00001 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=7f800000 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=40800000 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=40400000 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=3f000000 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=00000000 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=80000000 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=bf000000 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=c0400000 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=c0800000 result=7fc00001 errno=0 status=i
func=powf op1=ff800001 op2=ff800000 result=7fc00001 errno=0 status=i
func=powf op1=7fc00001 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=7fc00001 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=7fc00001 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=7fc00001 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=7fc00001 op2=7f800000 result=7fc00001 errno=0
func=powf op1=7fc00001 op2=40800000 result=7fc00001 errno=0
func=powf op1=7fc00001 op2=40400000 result=7fc00001 errno=0
func=powf op1=7fc00001 op2=3f000000 result=7fc00001 errno=0
func=powf op1=7fc00001 op2=00000000 result=3f800000 errno=0
func=powf op1=7fc00001 op2=80000000 result=3f800000 errno=0
func=powf op1=7fc00001 op2=bf000000 result=7fc00001 errno=0
func=powf op1=7fc00001 op2=c0400000 result=7fc00001 errno=0
func=powf op1=7fc00001 op2=c0800000 result=7fc00001 errno=0
func=powf op1=7fc00001 op2=ff800000 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=ffc00001 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=ffc00001 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=7f800000 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=40800000 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=40400000 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=3f000000 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=00000000 result=3f800000 errno=0
func=powf op1=ffc00001 op2=80000000 result=3f800000 errno=0
func=powf op1=ffc00001 op2=bf000000 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=c0400000 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=c0800000 result=7fc00001 errno=0
func=powf op1=ffc00001 op2=ff800000 result=7fc00001 errno=0
func=powf op1=7f800000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=7f800000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=7f800000 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=7f800000 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=7f800000 op2=7f800000 result=7f800000 errno=0
func=powf op1=7f800000 op2=40800000 result=7f800000 errno=0
func=powf op1=7f800000 op2=40400000 result=7f800000 errno=0
func=powf op1=7f800000 op2=3f000000 result=7f800000 errno=0
func=powf op1=7f800000 op2=00000001 result=7f800000 errno=0
func=powf op1=7f800000 op2=00000000 result=3f800000 errno=0
func=powf op1=7f800000 op2=80000000 result=3f800000 errno=0
func=powf op1=7f800000 op2=bf000000 result=00000000 errno=0
func=powf op1=7f800000 op2=c0400000 result=00000000 errno=0
func=powf op1=7f800000 op2=c0800000 result=00000000 errno=0
func=powf op1=7f800000 op2=ff800000 result=00000000 errno=0
func=powf op1=40800000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=40800000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=40800000 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=40800000 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=40800000 op2=7f800000 result=7f800000 errno=0
func=powf op1=40800000 op2=40800000 result=43800000 errno=0
func=powf op1=40800000 op2=40400000 result=42800000 errno=0
func=powf op1=40800000 op2=3f000000 result=40000000 errno=0
func=powf op1=40800000 op2=00000000 result=3f800000 errno=0
func=powf op1=40800000 op2=80000000 result=3f800000 errno=0
func=powf op1=40800000 op2=bf000000 result=3f000000 errno=0
func=powf op1=40800000 op2=c0400000 result=3c800000 errno=0
func=powf op1=40800000 op2=c0800000 result=3b800000 errno=0
func=powf op1=40800000 op2=ff800000 result=00000000 errno=0
func=powf op1=3f800000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=3f800000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=3f800000 op2=7fc00001 result=3f800000 errno=0
func=powf op1=3f800000 op2=ffc00001 result=3f800000 errno=0
func=powf op1=3f800000 op2=7f800000 result=3f800000 errno=0
func=powf op1=3f800000 op2=40800000 result=3f800000 errno=0
func=powf op1=3f800000 op2=40400000 result=3f800000 errno=0
func=powf op1=3f800000 op2=3f000000 result=3f800000 errno=0
func=powf op1=3f800000 op2=00000000 result=3f800000 errno=0
func=powf op1=3f800000 op2=80000000 result=3f800000 errno=0
func=powf op1=3f800000 op2=bf000000 result=3f800000 errno=0
func=powf op1=3f800000 op2=c0400000 result=3f800000 errno=0
func=powf op1=3f800000 op2=c0800000 result=3f800000 errno=0
func=powf op1=3f800000 op2=ff800000 result=3f800000 errno=0
func=powf op1=3e800000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=3e800000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=3e800000 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=3e800000 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=3e800000 op2=7f800000 result=00000000 errno=0
func=powf op1=3e800000 op2=40800000 result=3b800000 errno=0
func=powf op1=3e800000 op2=40400000 result=3c800000 errno=0
func=powf op1=3e800000 op2=3f000000 result=3f000000 errno=0
func=powf op1=3e800000 op2=00000000 result=3f800000 errno=0
func=powf op1=3e800000 op2=80000000 result=3f800000 errno=0
func=powf op1=3e800000 op2=bf000000 result=40000000 errno=0
func=powf op1=3e800000 op2=c0400000 result=42800000 errno=0
func=powf op1=3e800000 op2=c0800000 result=43800000 errno=0
func=powf op1=3e800000 op2=ff800000 result=7f800000 errno=0
func=powf op1=00000001 op2=bf800000 result=7f800000 errno=ERANGE status=ox
func=powf op1=00000000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=00000000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=00000000 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=00000000 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=00000000 op2=7f800000 result=00000000 errno=0
func=powf op1=00000000 op2=40800000 result=00000000 errno=0
func=powf op1=00000000 op2=40400000 result=00000000 errno=0
func=powf op1=00000000 op2=3f000000 result=00000000 errno=0
func=powf op1=00000000 op2=00000000 result=3f800000 errno=0
func=powf op1=00000000 op2=80000000 result=3f800000 errno=0
func=powf op1=00000000 op2=bf000000 result=7f800000 errno=ERANGE status=z
func=powf op1=00000000 op2=c0400000 result=7f800000 errno=ERANGE status=z
func=powf op1=00000000 op2=c0800000 result=7f800000 errno=ERANGE status=z
func=powf op1=00000000 op2=ff800000 result=7f800000 errno=ERANGE
func=powf op1=80000000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=80000000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=80000000 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=80000000 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=80000000 op2=7f800000 result=00000000 errno=0
func=powf op1=80000000 op2=40800000 result=00000000 errno=0
func=powf op1=80000000 op2=40400000 result=80000000 errno=0
func=powf op1=80000000 op2=3f000000 result=00000000 errno=0
func=powf op1=80000000 op2=00000000 result=3f800000 errno=0
func=powf op1=80000000 op2=80000000 result=3f800000 errno=0
func=powf op1=80000000 op2=bf000000 result=7f800000 errno=ERANGE status=z
func=powf op1=80000000 op2=c0400000 result=ff800000 errno=ERANGE status=z
func=powf op1=80000000 op2=c0800000 result=7f800000 errno=ERANGE status=z
func=powf op1=80000000 op2=ff800000 result=7f800000 errno=ERANGE
func=powf op1=be800000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=be800000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=be800000 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=be800000 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=be800000 op2=7f800000 result=00000000 errno=0
func=powf op1=be800000 op2=40800000 result=3b800000 errno=0
func=powf op1=be800000 op2=40400000 result=bc800000 errno=0
func=powf op1=be800000 op2=3f000000 result=7fc00001 errno=EDOM status=i
func=powf op1=be800000 op2=00000000 result=3f800000 errno=0
func=powf op1=be800000 op2=80000000 result=3f800000 errno=0
func=powf op1=be800000 op2=bf000000 result=7fc00001 errno=EDOM status=i
func=powf op1=be800000 op2=c0400000 result=c2800000 errno=0
func=powf op1=be800000 op2=c0800000 result=43800000 errno=0
func=powf op1=be800000 op2=ff800000 result=7f800000 errno=0
func=powf op1=bf800000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=bf800000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=bf800000 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=bf800000 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=bf800000 op2=7f800000 result=3f800000 errno=0
func=powf op1=bf800000 op2=40800000 result=3f800000 errno=0
func=powf op1=bf800000 op2=40400000 result=bf800000 errno=0
func=powf op1=bf800000 op2=3f000000 result=7fc00001 errno=EDOM status=i
func=powf op1=bf800000 op2=00000000 result=3f800000 errno=0
func=powf op1=bf800000 op2=80000000 result=3f800000 errno=0
func=powf op1=bf800000 op2=bf000000 result=7fc00001 errno=EDOM status=i
func=powf op1=bf800000 op2=c0400000 result=bf800000 errno=0
func=powf op1=bf800000 op2=c0800000 result=3f800000 errno=0
func=powf op1=bf800000 op2=ff800000 result=3f800000 errno=0
func=powf op1=c0800000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=c0800000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=c0800000 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=c0800000 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=c0800000 op2=7f800000 result=7f800000 errno=0
func=powf op1=c0800000 op2=40800000 result=43800000 errno=0
func=powf op1=c0800000 op2=40400000 result=c2800000 errno=0
func=powf op1=c0800000 op2=3f000000 result=7fc00001 errno=EDOM status=i
func=powf op1=c0800000 op2=00000000 result=3f800000 errno=0
func=powf op1=c0800000 op2=80000000 result=3f800000 errno=0
func=powf op1=c0800000 op2=bf000000 result=7fc00001 errno=EDOM status=i
func=powf op1=c0800000 op2=c0400000 result=bc800000 errno=0
func=powf op1=c0800000 op2=c0800000 result=3b800000 errno=0
func=powf op1=c0800000 op2=ff800000 result=00000000 errno=0
func=powf op1=ff800000 op2=7f800001 result=7fc00001 errno=0 status=i
func=powf op1=ff800000 op2=ff800001 result=7fc00001 errno=0 status=i
func=powf op1=ff800000 op2=7fc00001 result=7fc00001 errno=0
func=powf op1=ff800000 op2=ffc00001 result=7fc00001 errno=0
func=powf op1=ff800000 op2=7f800000 result=7f800000 errno=0
func=powf op1=ff800000 op2=40800000 result=7f800000 errno=0
func=powf op1=ff800000 op2=40400000 result=ff800000 errno=0
func=powf op1=ff800000 op2=3f000000 result=7f800000 errno=0
func=powf op1=ff800000 op2=00000000 result=3f800000 errno=0
func=powf op1=ff800000 op2=80000000 result=3f800000 errno=0
func=powf op1=ff800000 op2=bf000000 result=00000000 errno=0
func=powf op1=ff800000 op2=c0400000 result=80000000 errno=0
func=powf op1=ff800000 op2=c0800000 result=00000000 errno=0
func=powf op1=ff800000 op2=ff800000 result=00000000 errno=0


func=powf op1=36c27f9d op2=4109fa51 result=00000000 errno=ERANGE status=ux
func=powf op1=351738cd op2=c0c55691 result=7f800000 errno=ERANGE status=ox
func=powf op1=42836035 op2=41a99f40 result=7f800000 errno=ERANGE status=ox
func=powf op1=32bd53f3 op2=40bcba58 result=00000000 errno=ERANGE status=ux
func=powf op1=32dc5bff op2=40be62ea result=00000000 errno=ERANGE status=ux
func=powf op1=3a8a3f66 op2=4172bd43 result=00000000 errno=ERANGE status=ux
func=powf op1=28f0e770 op2=c035b4ca result=7f800000 errno=ERANGE status=ox
func=powf op1=40886699 op2=c28f703a result=00000000 errno=ERANGE status=ux
func=powf op1=414bd593 op2=c22370cf result=00000000 errno=ERANGE status=ux
func=powf op1=3a2f1163 op2=c1422d45 result=7f800000 errno=ERANGE status=ox
func=powf op1=434f5cf3 op2=41851272 result=7f800000 errno=ERANGE status=ox
func=powf op1=2e0e27a4 op2=c06b13f5 result=7f800000 errno=ERANGE status=ox
func=powf op1=39aef7a6 op2=414fd60a result=00000000 errno=ERANGE status=ux
func=powf op1=21c80729 op2=c00a04ab result=7f800000 errno=ERANGE status=ox
func=powf op1=42455a4b op2=c1d55905 result=00000000 errno=ERANGE status=ux
func=powf op1=2d173e0b op2=c05ee797 result=7f800000 errno=ERANGE status=ox
func=powf op1=452edf9a op2=4132dd7f result=7f800000 errno=ERANGE status=ox
func=powf op1=406bf67b op2=c29f5f12 result=00000000 errno=ERANGE status=ux
func=powf op1=2d82a6fc op2=4085779e result=00000000 errno=ERANGE status=ux
func=powf op1=4551f827 op2=41304516 result=7f800000 errno=ERANGE status=ox
func=powf op1=3a917c51 op2=41726c0a result=00000001.37f errno=0 status=ux
; iso c allows both errno=ERANGE and errno=0
;func=powf op1=3b19bbaa op2=4188e6fb result=00000000.b5f errno=0 status=ux
;func=powf op1=4088bd18 op2=c28ef056 result=00000000.986 errno=0 status=ux
func=powf op1=3f7ffd76 op2=4a09221e result=00aa9d24.3ad error=0

func=powf op1=007fffff op2=bf000001 result=5f00002c.2b2 error=0
func=powf op1=000007ff op2=bf000001 result=62000830.96f error=0
func=powf op1=007fffff op2=80800001 result=3f800000.000 error=0
func=powf op1=00000000 op2=800007ff result=7f800000 errno=ERANGE status=z
func=powf op1=00000000 op2=000007ff result=00000000 error=0
func=powf op1=bf800000 op2=ff7fffff result=3f800000 error=0
func=powf op1=2e4e4f30 op2=406b0dc2 result=007e9c59.eb4 errno=0 status=u

; SDCOMP-25549: ensure the biggest overflow case possible is not
; mishandled. Also check the analogous underflow, and also ensure that
; our massive-overflow checks do not affect numbers _just within_ the
; range.
func=powf op1=7f7fffff op2=7f7fffff result=7f800000 error=overflow
func=powf op1=7f7fffff op2=ff7fffff result=00000000 error=underflow
func=powf op1=54cb3000 op2=403fffff result=7f7fffb2.a95 error=0
