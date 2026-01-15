//===-- orc_rt_elfnix_tls_systemz.s -------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
//===----------------------------------------------------------------------===//

// The special thing about the s390 TLS ABI is that we do not have the
// standard __tls_get_addr function but the __tls_get_offset function
// which differs in two important aspects:
// 1) __tls_get_offset gets a got offset instead of a pointer to the
//    tls_index structure
// 2) __tls_get_offset returns the offset of the requested variable to
//    the thread descriptor instead of a pointer to the variable.

// The content of this file is systemz-only

#if defined(__s390x__)

	.text
	// returns offset of TLV from TP in %r2.
	.globl	___orc_rt_elfnix_tls_get_offset
___orc_rt_elfnix_tls_get_offset:
	stmg	%r14, %r15, 112(%r15)
	aghi	%r15, -160
	// Pass pointer to tls_index.
	la 	%r2, 0(%r2, %r12)
	brasl	%r14, __orc_rt_elfnix_tls_get_addr_impl
	// Return offset from TP.
	ear 	%r0, %a0
	sllg	%r0, %r0, 32
	ear 	%r0, %a1
	sgr 	%r2, %r0
	lmg 	%r14, %r15, 272(%r15)
	br	%r14

#endif // defined(__s390x__)
