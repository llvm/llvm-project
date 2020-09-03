; log2f.tst - Directed test cases for log2f
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func=log2f op1=7fc00001 result=7fc00001 errno=0
func=log2f op1=ffc00001 result=7fc00001 errno=0
func=log2f op1=7f800001 result=7fc00001 errno=0 status=i
func=log2f op1=ff800001 result=7fc00001 errno=0 status=i
func=log2f op1=ff810000 result=7fc00001 errno=0 status=i
func=log2f op1=7f800000 result=7f800000 errno=0
func=log2f op1=ff800000 result=7fc00001 errno=EDOM status=i
func=log2f op1=3f800000 result=00000000 errno=0
func=log2f op1=00000000 result=ff800000 errno=ERANGE status=z
func=log2f op1=80000000 result=ff800000 errno=ERANGE status=z
func=log2f op1=80000001 result=7fc00001 errno=EDOM status=i

func=log2f op1=3f7d70a4 result=bc6d8f8b.7d4 error=0
func=log2f op1=3f604189 result=be4394c8.395 error=0
func=log2f op1=3f278034 result=bf1caa73.88e error=0
func=log2f op1=3edd3c36 result=bf9af3b9.619 error=0
func=log2f op1=3e61259a result=c00bdb95.650 error=0
func=log2f op1=3f8147ae result=3c6b3267.d6a error=0
func=log2f op1=3f8fbe77 result=3e2b5fe2.a1c error=0
func=log2f op1=3fac3eea result=3edb4d5e.1fc error=0
func=log2f op1=3fd6e632 result=3f3f5d3a.827 error=0
func=log2f op1=40070838 result=3f89e055.a0a error=0
