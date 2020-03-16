; exp2f.tst - Directed test cases for exp2f
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func=exp2f op1=7fc00001 result=7fc00001 errno=0
func=exp2f op1=ffc00001 result=7fc00001 errno=0
func=exp2f op1=7f800001 result=7fc00001 errno=0 status=i
func=exp2f op1=ff800001 result=7fc00001 errno=0 status=i
func=exp2f op1=7f800000 result=7f800000 errno=0
func=exp2f op1=7f7fffff result=7f800000 errno=ERANGE status=ox
func=exp2f op1=ff800000 result=00000000 errno=0
func=exp2f op1=ff7fffff result=00000000 errno=ERANGE status=ux
func=exp2f op1=00000000 result=3f800000 errno=0
func=exp2f op1=80000000 result=3f800000 errno=0
func=exp2f op1=42fa0001 result=7e00002c.5c8 errno=0
func=exp2f op1=42ffffff result=7f7fffa7.470 errno=0
func=exp2f op1=43000000 result=7f800000 errno=ERANGE status=ox
func=exp2f op1=43000001 result=7f800000 errno=ERANGE status=ox
func=exp2f op1=c2fa0001 result=00ffffa7.470 errno=0
func=exp2f op1=c2fc0000 result=00800000 errno=0
func=exp2f op1=c2fc0001 result=007fffd3.a38 errno=0 status=ux
func=exp2f op1=c3150000 result=00000001 errno=0
func=exp2f op1=c3158000 result=00000000.800 errno=ERANGE status=ux
func=exp2f op1=c3165432 result=00000000.4bd errno=ERANGE status=ux
