/*
 * Copyright (c) 1989, 1993
 *	The Regents of the University of California.  All rights reserved.
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

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)getttyent.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */

#include <ttyent.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <ctype.h>
#include <string.h>

#define flockfile(s) _IO_flockfile (s)
#define funlockfile(s) _IO_funlockfile (s)

static char zapchar;
static FILE *tf;

struct ttyent *
__getttynam (const char *tty)
{
	struct ttyent *t;

	__setttyent();
	while ((t = __getttyent()))
		if (!strcmp(tty, t->ty_name))
			break;
	__endttyent();
	return (t);
}
weak_alias (__getttynam, getttynam)

static char *skip (char *) __THROW;
static char *value (char *) __THROW;

struct ttyent *
__getttyent (void)
{
	static struct ttyent tty;
	int c;
	char *p;
#define	MAXLINELENGTH	100
	static char line[MAXLINELENGTH];

	if (!tf && !__setttyent())
		return (NULL);
	flockfile (tf);
	for (;;) {
		if (!__fgets_unlocked(p = line, sizeof(line), tf)) {
			funlockfile (tf);
			return (NULL);
		}
		/* skip lines that are too big */
		if (!strchr (p, '\n')) {
			while ((c = __getc_unlocked(tf)) != '\n' && c != EOF)
				;
			continue;
		}
		while (isspace(*p))
			++p;
		if (*p && *p != '#')
			break;
	}

	zapchar = 0;
	tty.ty_name = p;
	p = skip(p);
	if (!*(tty.ty_getty = p))
		tty.ty_getty = tty.ty_type = NULL;
	else {
		p = skip(p);
		if (!*(tty.ty_type = p))
			tty.ty_type = NULL;
		else
			p = skip(p);
	}
	tty.ty_status = 0;
	tty.ty_window = NULL;

#define	scmp(e)	!strncmp(p, e, sizeof(e) - 1) && isspace(p[sizeof(e) - 1])
#define	vcmp(e)	!strncmp(p, e, sizeof(e) - 1) && p[sizeof(e) - 1] == '='
	for (; *p; p = skip(p)) {
		if (scmp(_TTYS_OFF))
			tty.ty_status &= ~TTY_ON;
		else if (scmp(_TTYS_ON))
			tty.ty_status |= TTY_ON;
		else if (scmp(_TTYS_SECURE))
			tty.ty_status |= TTY_SECURE;
		else if (vcmp(_TTYS_WINDOW))
			tty.ty_window = value(p);
		else
			break;
	}
	/* We can release the lock only here since `zapchar' is global.  */
	funlockfile(tf);

	if (zapchar == '#' || *p == '#')
		while ((c = *++p) == ' ' || c == '\t')
			;
	tty.ty_comment = p;
	if (*p == 0)
		tty.ty_comment = 0;
	if ((p = strchr (p, '\n')))
		*p = '\0';
	return (&tty);
}
libc_hidden_def (__getttyent)
weak_alias (__getttyent, getttyent)

#define	QUOTED	1

/*
 * Skip over the current field, removing quotes, and return a pointer to
 * the next field.
 */
static char *
skip (char *p)
{
	char *t;
	int c, q;

	for (q = 0, t = p; (c = *p) != '\0'; p++) {
		if (c == '"') {
			q ^= QUOTED;	/* obscure, but nice */
			continue;
		}
		if (q == QUOTED && *p == '\\' && *(p+1) == '"')
			p++;
		*t++ = *p;
		if (q == QUOTED)
			continue;
		if (c == '#') {
			zapchar = c;
			*p = 0;
			break;
		}
		if (c == '\t' || c == ' ' || c == '\n') {
			zapchar = c;
			*p++ = 0;
			while ((c = *p) == '\t' || c == ' ' || c == '\n')
				p++;
			break;
		}
	}
	*--t = '\0';
	return (p);
}

static char *
value (char *p)
{

	return ((p = strchr (p, '=')) ? ++p : NULL);
}

int
__setttyent (void)
{

	if (tf) {
		(void)rewind(tf);
		return (1);
	} else if ((tf = fopen(_PATH_TTYS, "rce"))) {
		/* We do the locking ourselves.  */
		__fsetlocking (tf, FSETLOCKING_BYCALLER);
		return (1);
	}
	return (0);
}
libc_hidden_def (__setttyent)
weak_alias (__setttyent, setttyent)

int
__endttyent (void)
{
	int rval;

	if (tf) {
		rval = !(fclose(tf) == EOF);
		tf = NULL;
		return (rval);
	}
	return (1);
}
libc_hidden_def (__endttyent)
weak_alias (__endttyent, endttyent)
