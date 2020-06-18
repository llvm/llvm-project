; expf.tst - Directed test cases for expf
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func=expf op1=7fc00001 result=7fc00001 errno=0
func=expf op1=ffc00001 result=7fc00001 errno=0
func=expf op1=7f800001 result=7fc00001 errno=0 status=i
func=expf op1=ff800001 result=7fc00001 errno=0 status=i
func=expf op1=7f800000 result=7f800000 errno=0
func=expf op1=7f7fffff result=7f800000 errno=ERANGE status=ox
func=expf op1=ff800000 result=00000000 errno=0
func=expf op1=ff7fffff result=00000000 errno=ERANGE status=ux
func=expf op1=00000000 result=3f800000 errno=0
func=expf op1=80000000 result=3f800000 errno=0
func=expf op1=42affff8 result=7ef87ed4.e0c errno=0
func=expf op1=42b00008 result=7ef88698.f67 errno=0
func=expf op1=42cffff8 result=7f800000 errno=ERANGE status=ox
func=expf op1=42d00008 result=7f800000 errno=ERANGE status=ox
func=expf op1=c2affff8 result=0041eecc.041 errno=0 status=ux
func=expf op1=c2b00008 result=0041ecbc.95e errno=0 status=ux
func=expf op1=c2cffff8 result=00000000 errno=ERANGE status=ux
func=expf op1=c2d00008 result=00000000 errno=ERANGE status=ux
