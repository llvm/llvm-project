/*
 * Copyright (c) 1985, 1993
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
static char sccsid[] = "@(#)getusershell.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */

#include <sys/param.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>
#include <paths.h>

/*
 * Local shells should NOT be added here.  They should be added in
 * /etc/shells.
 */

/* NB: we do not initialize okshells here.  The initialization needs
   relocations.  These interfaces are used so rarely that this is not
   justified.  Instead explicitly initialize the array when it is
   used.  */
#if 0
static const char *const okshells[] = { _PATH_BSHELL, _PATH_CSHELL, NULL };
#else
static const char *okshells[3];
#endif
static char **curshell, **shells, *strings;
static char **initshells (void) __THROW;

/*
 * Get a list of shells from _PATH_SHELLS, if it exists.
 */
char *
getusershell (void)
{
	char *ret;

	if (curshell == NULL)
		curshell = initshells();
	ret = *curshell;
	if (ret != NULL)
		curshell++;
	return (ret);
}

void
endusershell (void)
{

	free(shells);
	shells = NULL;
	free(strings);
	strings = NULL;
	curshell = NULL;
}

void
setusershell (void)
{

	curshell = initshells();
}

static char **
initshells (void)
{
	char **sp, *cp;
	FILE *fp;
	struct stat64 statb;
	size_t flen;

	free(shells);
	shells = NULL;
	free(strings);
	strings = NULL;
	if ((fp = fopen(_PATH_SHELLS, "rce")) == NULL)
		goto init_okshells_noclose;
	if (__fstat64(fileno(fp), &statb) == -1) {
	init_okshells:
		(void)fclose(fp);
	init_okshells_noclose:
		okshells[0] = _PATH_BSHELL;
		okshells[1] = _PATH_CSHELL;
		return (char **) okshells;
	}
	if (statb.st_size > ~(size_t)0 / sizeof (char *) * 3)
		goto init_okshells;
	flen = statb.st_size + 3;
	if ((strings = malloc(flen)) == NULL)
		goto init_okshells;
	shells = malloc(statb.st_size / 3 * sizeof (char *));
	if (shells == NULL) {
		free(strings);
		strings = NULL;
		goto init_okshells;
	}
	sp = shells;
	cp = strings;
	while (fgets_unlocked(cp, flen - (cp - strings), fp) != NULL) {
		while (*cp != '#' && *cp != '/' && *cp != '\0')
			cp++;
		if (*cp == '#' || *cp == '\0' || cp[1] == '\0')
			continue;
		*sp++ = cp;
		while (!isspace(*cp) && *cp != '#' && *cp != '\0')
			cp++;
		*cp++ = '\0';
	}
	*sp = NULL;
	(void)fclose(fp);
	return (shells);
}
