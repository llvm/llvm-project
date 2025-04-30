/*
 * pmap_clnt.h
 * Supplies C routines to get to portmap services.
 *
 * Copyright (c) 2010, Oracle America, Inc.
 *
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

#ifndef _RPC_PMAP_CLNT_H
#define _RPC_PMAP_CLNT_H	1

#include <features.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
#include <rpc/clnt.h>

__BEGIN_DECLS

typedef bool_t (*resultproc_t) (caddr_t __resp, struct sockaddr_in *__raddr);

/*
 * Usage:
 *	success = pmap_set(program, version, protocol, port);
 *	success = pmap_unset(program, version);
 *	port = pmap_getport(address, program, version, protocol);
 *	head = pmap_getmaps(address);
 *	clnt_stat = pmap_rmtcall(address, program, version, procedure,
 *		xdrargs, argsp, xdrres, resp, tout, port_ptr)
 *		(works for udp only.)
 * 	clnt_stat = clnt_broadcast(program, version, procedure,
 *		xdrargs, argsp,	xdrres, resp, eachresult)
 *		(like pmap_rmtcall, except the call is broadcasted to all
 *		locally connected nets.  For each valid response received,
 *		the procedure eachresult is called.  Its form is:
 *	done = eachresult(resp, raddr)
 *		bool_t done;
 *		caddr_t resp;
 *		struct sockaddr_in raddr;
 *		where resp points to the results of the call and raddr is the
 *		address if the responder to the broadcast.
 */

extern bool_t pmap_set (const u_long __program, const u_long __vers,
			int __protocol, u_short __port) __THROW;
extern bool_t pmap_unset (const u_long __program, const u_long __vers)
     __THROW;
extern struct pmaplist *pmap_getmaps (struct sockaddr_in *__address) __THROW;
extern enum clnt_stat pmap_rmtcall (struct sockaddr_in *__addr,
				    const u_long __prog,
				    const u_long __vers,
				    const u_long __proc,
				    xdrproc_t __xdrargs,
				    caddr_t __argsp, xdrproc_t __xdrres,
				    caddr_t __resp, struct timeval __tout,
				    u_long *__port_ptr) __THROW;
extern enum clnt_stat clnt_broadcast (const u_long __prog,
				      const u_long __vers,
				      const u_long __proc, xdrproc_t __xargs,
				      caddr_t __argsp, xdrproc_t __xresults,
				      caddr_t __resultsp,
				      resultproc_t __eachresult) __THROW;
extern u_short pmap_getport (struct sockaddr_in *__address,
			     const u_long __program,
			     const u_long __version, u_int __protocol)
     __THROW;

__END_DECLS

#endif /* rpc/pmap_clnt.h */
