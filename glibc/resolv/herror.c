/*
 * Copyright (c) 1987, 1993
 *    The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Portions Copyright (c) 1996-1999 by Internet Software Consortium.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND INTERNET SOFTWARE CONSORTIUM DISCLAIMS
 * ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL INTERNET SOFTWARE
 * CONSORTIUM BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/uio.h>

#include <netinet/in.h>
#include <arpa/nameser.h>

#include <netdb.h>
#include <resolv.h>
#include <string.h>
#include <unistd.h>

#include <libintl.h>
#include <not-cancel.h>

const char *const h_errlist[] = {
	N_("Resolver Error 0 (no error)"),
	N_("Unknown host"),			/* 1 HOST_NOT_FOUND */
	N_("Host name lookup failure"),		/* 2 TRY_AGAIN */
	N_("Unknown server error"),		/* 3 NO_RECOVERY */
	N_("No address associated with name"),	/* 4 NO_ADDRESS */
};
const int	h_nerr = { sizeof h_errlist / sizeof h_errlist[0] };

/*
 * herror --
 *	print the error indicated by the h_errno value.
 */
void
herror(const char *s) {
	struct iovec iov[4], *v = iov;

	if (s != NULL && *s != '\0') {
		v->iov_base = (/*noconst*/ char *)s;
		v->iov_len = strlen(s);
		v++;
		v->iov_base = (char *) ": ";
		v->iov_len = 2;
		v++;
	}
	v->iov_base = (char *)hstrerror(h_errno);
	v->iov_len = strlen(v->iov_base);
	v++;
	v->iov_base = (char *) "\n";
	v->iov_len = 1;
	__writev_nocancel_nostatus(STDERR_FILENO, iov, (v - iov) + 1);
}

/*
 * hstrerror --
 *	return the string associated with a given "host" errno value.
 */
const char *
hstrerror(int err) {
	if (err < 0)
		return _("Resolver internal error");
	else if (err < h_nerr)
		return _(h_errlist[err]);
	return _("Unknown resolver error");
}
libc_hidden_def (hstrerror)
