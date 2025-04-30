/*
 * svc_simple.c
 * Simplified front end to rpc.
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

#include <stdio.h>
#include <string.h>
#include <libintl.h>
#include <unistd.h>
#include <rpc/rpc.h>
#include <rpc/pmap_clnt.h>
#include <sys/socket.h>
#include <netdb.h>

#include <wchar.h>
#include <libio/iolibio.h>
#include <shlib-compat.h>

struct proglst_
  {
    char *(*p_progname) (char *);
    int p_prognum;
    int p_procnum;
    xdrproc_t p_inproc, p_outproc;
    struct proglst_ *p_nxt;
  };
#define proglst RPC_THREAD_VARIABLE(svcsimple_proglst_s)


static void universal (struct svc_req *rqstp, SVCXPRT *transp_s);
#define transp RPC_THREAD_VARIABLE(svcsimple_transp_s)

int
__registerrpc (u_long prognum, u_long versnum, u_long procnum,
	       char *(*progname) (char *), xdrproc_t inproc, xdrproc_t outproc)
{
  struct proglst_ *pl;
  char *buf;

  if (procnum == NULLPROC)
    {

      if (__asprintf (&buf, _("can't reassign procedure number %ld\n"),
		      NULLPROC) < 0)
	buf = NULL;
      goto err_out;
    }
  if (transp == 0)
    {
      transp = svcudp_create (RPC_ANYSOCK);
      if (transp == NULL)
	{
	  buf = __strdup (_("couldn't create an rpc server\n"));
	  goto err_out;
	}
    }
  (void) pmap_unset ((u_long) prognum, (u_long) versnum);
  if (!svc_register (transp, (u_long) prognum, (u_long) versnum,
		     universal, IPPROTO_UDP))
    {
      if (__asprintf (&buf, _("couldn't register prog %ld vers %ld\n"),
		      prognum, versnum) < 0)
	buf = NULL;
      goto err_out;
    }
  pl = (struct proglst_ *) malloc (sizeof (struct proglst_));
  if (pl == NULL)
    {
      buf = __strdup (_("registerrpc: out of memory\n"));
      goto err_out;
    }
  pl->p_progname = progname;
  pl->p_prognum = prognum;
  pl->p_procnum = procnum;
  pl->p_inproc = inproc;
  pl->p_outproc = outproc;
  pl->p_nxt = proglst;
  proglst = pl;
  return 0;

 err_out:
  if (buf == NULL)
    return -1;
  (void) __fxprintf (NULL, "%s", buf);
  free (buf);
  return -1;
}

libc_sunrpc_symbol (__registerrpc, registerrpc, GLIBC_2_0)


static void
universal (struct svc_req *rqstp, SVCXPRT *transp_l)
{
  int prog, proc;
  char *outdata;
  char xdrbuf[UDPMSGSIZE];
  struct proglst_ *pl;
  char *buf = NULL;

  /*
   * enforce "procnum 0 is echo" convention
   */
  if (rqstp->rq_proc == NULLPROC)
    {
      if (svc_sendreply (transp_l, (xdrproc_t)xdr_void,
			 (char *) NULL) == FALSE)
	{
	  __write (STDERR_FILENO, "xxx\n", 4);
	  exit (1);
	}
      return;
    }
  prog = rqstp->rq_prog;
  proc = rqstp->rq_proc;
  for (pl = proglst; pl != NULL; pl = pl->p_nxt)
    if (pl->p_prognum == prog && pl->p_procnum == proc)
      {
	/* decode arguments into a CLEAN buffer */
	memset (xdrbuf, 0, sizeof (xdrbuf));	/* required ! */
	if (!svc_getargs (transp_l, pl->p_inproc, xdrbuf))
	  {
	    svcerr_decode (transp_l);
	    return;
	  }
	outdata = (*(pl->p_progname)) (xdrbuf);
	if (outdata == NULL && pl->p_outproc != (xdrproc_t)xdr_void)
	  /* there was an error */
	  return;
	if (!svc_sendreply (transp_l, pl->p_outproc, outdata))
	  {
	    if (__asprintf (&buf, _("trouble replying to prog %d\n"),
			    pl->p_prognum) < 0)
	      buf = NULL;
	    goto err_out2;
	  }
	/* free the decoded arguments */
	(void) svc_freeargs (transp_l, pl->p_inproc, xdrbuf);
	return;
      }
  if (__asprintf (&buf, _("never registered prog %d\n"), prog) < 0)
    buf = NULL;
 err_out2:
  if (buf == NULL)
    exit (1);
  __fxprintf (NULL, "%s", buf);
  free (buf);
  exit (1);
}
