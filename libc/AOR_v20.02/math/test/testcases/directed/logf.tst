; logf.tst - Directed test cases for logf
;
; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func=logf op1=7fc00001 result=7fc00001 errno=0
func=logf op1=ffc00001 result=7fc00001 errno=0
func=logf op1=7f800001 result=7fc00001 errno=0 status=i
func=logf op1=ff800001 result=7fc00001 errno=0 status=i
func=logf op1=ff810000 result=7fc00001 errno=0 status=i
func=logf op1=7f800000 result=7f800000 errno=0
func=logf op1=ff800000 result=7fc00001 errno=EDOM status=i
func=logf op1=3f800000 result=00000000 errno=0
func=logf op1=00000000 result=ff800000 errno=ERANGE status=z
func=logf op1=80000000 result=ff800000 errno=ERANGE status=z
func=logf op1=80000001 result=7fc00001 errno=EDOM status=i

; Directed tests for the special-case handling of log of things
; very near 1
func=logf op1=3f781e49 result=bd0016d9.4ae error=0
func=logf op1=3f78e602 result=bce675e5.f31 error=0
func=logf op1=3f844a18 result=3d07030e.ae1 error=0
func=logf op1=3f79b55b result=bccbd88a.6cb error=0
func=logf op1=3f7e2f5f result=bbe92452.74a error=0
func=logf op1=3f7f1c03 result=bb6462c1.c2c error=0
func=logf op1=3f78b213 result=bced23e2.f56 error=0
func=logf op1=3f87d5c0 result=3d735847.b7a error=0
func=logf op1=3f7fa6ad result=bab2c532.12d error=0
func=logf op1=3f87c06a result=3d70d4b6.b5e error=0
func=logf op1=3f79cf30 result=bcc88942.6e9 error=0
func=logf op1=3f794c77 result=bcd94c6f.b1e error=0
func=logf op1=3f835655 result=3cd2d8a0.0bf error=0
func=logf op1=3f81b5c0 result=3c596d08.520 error=0
func=logf op1=3f805e2f result=3b3c18d4.d2b error=0
func=logf op1=3f7aa609 result=bcad0f90.fdb error=0
func=logf op1=3f7a9091 result=bcafcd59.f83 error=0
func=logf op1=3f7a7475 result=bcb36490.a0f error=0
func=logf op1=3f823417 result=3c8bd287.fa6 error=0
func=logf op1=3f7fbcc3 result=ba868bac.14c error=0
func=logf op1=3f805fc9 result=3b3f4a76.169 error=0
func=logf op1=3f833d43 result=3cccbc4f.cb7 error=0
func=logf op1=3f7cb1de result=bc54e91e.6b5 error=0
func=logf op1=3f7f2793 result=bb58c8af.bfc error=0
func=logf op1=3f7bb8c3 result=bc8a0fc9.93c error=0
func=logf op1=3f81d349 result=3c67fe09.42e error=0
func=logf op1=3f7c254d result=bc788cf4.610 error=0
func=logf op1=3f7f789d result=bb0786d9.6c6 error=0
func=logf op1=3f7ed1f2 result=bb97605f.963 error=0
func=logf op1=3f826067 result=3c96b4af.5e1 error=0
func=logf op1=3f821a68 result=3c8581f9.dac error=0
func=logf op1=3f864e1a result=3d44f368.e66 error=0
func=logf op1=3f7fea3d result=b9ae1f66.b58 error=0
func=logf op1=3f7cf4f5 result=bc43ed76.1c5 error=0
func=logf op1=3f84c223 result=3d15814e.36d error=0
func=logf op1=3f7dae6d result=bc1511d5.0aa error=0
func=logf op1=3f7c0a3c result=bc7f6c0d.758 error=0
func=logf op1=3f858b22 result=3d2da861.f36 error=0
func=logf op1=3f85d7c7 result=3d36d490.ee9 error=0
func=logf op1=3f7f2109 result=bb5f5851.2ed error=0
func=logf op1=3f83809c result=3cdd23f7.6b1 error=0
func=logf op1=3f83d96e result=3cf2b9c8.0b1 error=0
func=logf op1=3f86ca84 result=3d53bee8.53f error=0
func=logf op1=3f83548e result=3cd269c3.39d error=0
func=logf op1=3f7c199c result=bc7b84b6.0da error=0
func=logf op1=3f83133f result=3cc27c0a.9dd error=0
func=logf op1=3f7c97b4 result=bc5b89dd.399 error=0
func=logf op1=3f810bc1 result=3c05553c.011 error=0
func=logf op1=3f7dadb8 result=bc153f7e.fbb error=0
func=logf op1=3f87be56 result=3d709602.538 error=0
