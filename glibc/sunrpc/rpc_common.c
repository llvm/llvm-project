/*
 * Copyright (c) 2010, Oracle America, Inc.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the "Oracle America, Inc." nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 *   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <rpc/rpc.h>
#include <shlib-compat.h>

#undef svc_fdset
#undef rpc_createerr
#undef svc_pollfd
#undef svc_max_pollfd

/*
 * This file should only contain common data (global data) that is exported
 * by public interfaces
 */
/* We are very tricky here.  We want to have _null_auth in a read-only
   section but we cannot add const to the type because this isn't how
   the variable is declared.  So we use the section attribute.  */
struct opaque_auth _null_auth;
libc_hidden_nolink_sunrpc (_null_auth, GLIBC_2_0)

fd_set svc_fdset;
struct rpc_createerr rpc_createerr;
struct pollfd *svc_pollfd;
int svc_max_pollfd;
#ifdef SHARED
# ifndef EXPORT_RPC_SYMBOLS
compat_symbol (libc, svc_fdset, svc_fdset, GLIBC_2_0);
compat_symbol (libc, rpc_createerr, rpc_createerr, GLIBC_2_0);
compat_symbol (libc, svc_pollfd, svc_pollfd, GLIBC_2_2);
compat_symbol (libc, svc_max_pollfd, svc_max_pollfd, GLIBC_2_2);
# endif
#endif
