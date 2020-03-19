; sinf.tst - Directed test cases for SP sine
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


func=sinf op1=7fc00001 result=7fc00001 errno=0
func=sinf op1=ffc00001 result=7fc00001 errno=0
func=sinf op1=7f800001 result=7fc00001 errno=0 status=i
func=sinf op1=ff800001 result=7fc00001 errno=0 status=i
func=sinf op1=7f800000 result=7fc00001 errno=EDOM status=i
func=sinf op1=ff800000 result=7fc00001 errno=EDOM status=i
func=sinf op1=00000000 result=00000000 errno=0
func=sinf op1=80000000 result=80000000 errno=0
; Directed test for a failure I found while developing mathbench
func=sinf op1=c70d39a1 result=be37fad5.7ed errno=0
; SDCOMP-26094: check sinf in the cases for which the range reducer
; returns values furthest beyond its nominal upper bound of pi/4.
func=sinf op1=46427f1b result=3f352d80.f9b error=0
func=sinf op1=4647e568 result=3f352da9.7be error=0
func=sinf op1=46428bac result=bf352dea.924 error=0
func=sinf op1=4647f1f9 result=bf352e13.146 error=0
func=sinf op1=4647fe8a result=3f352e7c.ac9 error=0
func=sinf op1=45d8d7f1 result=3f35097b.cb0 error=0
func=sinf op1=45d371a4 result=bf350990.102 error=0
func=sinf op1=45ce0b57 result=3f3509a4.554 error=0
func=sinf op1=45d35882 result=3f3509f9.bdb error=0
func=sinf op1=45cdf235 result=bf350a0e.02c error=0
