/*
 * Copyright (C) 1998 WIDE Project.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
/*
 * Copyright (c) 1983, 1993, 1994
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

#include <sys/param.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include <netinet/in.h>
#include <arpa/inet.h>

#include <alloca.h>
#include <signal.h>
#include <fcntl.h>
#include <netdb.h>
#include <unistd.h>
#include <pwd.h>
#include <errno.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <ctype.h>
#include <string.h>
#include <libintl.h>
#include <stdlib.h>
#include <wchar.h>
#include <sys/uio.h>
#include <sigsetops.h>


int __ivaliduser (FILE *, uint32_t, const char *, const char *);
static int __validuser2_sa (FILE *, struct sockaddr *, size_t,
			    const char *, const char *, const char *);
static int ruserok2_sa (struct sockaddr *ra, size_t ralen,
			int superuser, const char *ruser,
			const char *luser, const char *rhost);
static int ruserok_sa (struct sockaddr *ra, size_t ralen,
			int superuser, const char *ruser,
			const char *luser);
int iruserok_af (const void *raddr, int superuser, const char *ruser,
		 const char *luser, sa_family_t af);
int iruserok (uint32_t raddr, int superuser, const char *ruser,
	      const char *luser);

libc_hidden_proto (iruserok_af)

libc_freeres_ptr(static char *ahostbuf);

int
rcmd_af (char **ahost, u_short rport, const char *locuser, const char *remuser,
	 const char *cmd, int *fd2p, sa_family_t af)
{
	char paddr[INET6_ADDRSTRLEN];
	struct addrinfo hints, *res, *ai;
	union
	{
		struct sockaddr sa;
		struct sockaddr_storage ss;
		struct sockaddr_in sin;
		struct sockaddr_in6 sin6;
	} from;
	struct pollfd pfd[2];
	sigset_t mask, omask;

	pid_t pid;
	int s, lport, timo, error;
	char c;
	int refused;
	char num[8];
	ssize_t n;

	if (af != AF_INET && af != AF_INET6 && af != AF_UNSPEC)
	  {
	    __set_errno (EAFNOSUPPORT);
	    return -1;
	  }

	pid = __getpid();

	memset(&hints, '\0', sizeof(hints));
	hints.ai_flags = AI_CANONNAME;
	hints.ai_family = af;
	hints.ai_socktype = SOCK_STREAM;
	(void)__snprintf(num, sizeof(num), "%d", ntohs(rport));
	error = getaddrinfo(*ahost, num, &hints, &res);
	if (error) {
		if (error == EAI_NONAME && *ahost != NULL)
			__fxprintf(NULL, "%s: Unknown host\n", *ahost);
		else
			__fxprintf(NULL, "rcmd: getaddrinfo: %s\n",
				   gai_strerror(error));

		return -1;
	}

	pfd[0].events = POLLIN;
	pfd[1].events = POLLIN;

	if (res->ai_canonname){
		free (ahostbuf);
		ahostbuf = __strdup (res->ai_canonname);
		if (ahostbuf == NULL) {
			freeaddrinfo (res);
			__fxprintf(NULL, "%s",
				   _("rcmd: Cannot allocate memory\n"));
			return -1;
		}
		*ahost = ahostbuf;
	} else
		*ahost = NULL;
	ai = res;
	refused = 0;
	__sigemptyset(&mask);
	__sigaddset(&mask, SIGURG);
	__sigprocmask (SIG_BLOCK, &mask, &omask);
	for (timo = 1, lport = IPPORT_RESERVED - 1;;) {
		char errbuf[200];

		s = rresvport_af(&lport, ai->ai_family);
		if (s < 0) {
			if (errno == EAGAIN)
				__fxprintf(NULL, "%s", _("\
rcmd: socket: All ports in use\n"));
			else
				__fxprintf(NULL, "rcmd: socket: %m\n");

			__sigprocmask (SIG_SETMASK, &omask, 0);
			freeaddrinfo(res);
			return -1;
		}
		__fcntl(s, F_SETOWN, pid);
		if (__connect(s, ai->ai_addr, ai->ai_addrlen) >= 0)
			break;
		(void)__close(s);
		if (errno == EADDRINUSE) {
			lport--;
			continue;
		}
		if (errno == ECONNREFUSED)
			refused = 1;
		if (ai->ai_next != NULL) {
			int oerrno = errno;
			char *buf = NULL;

			getnameinfo(ai->ai_addr, ai->ai_addrlen,
				    paddr, sizeof(paddr),
				    NULL, 0,
				    NI_NUMERICHOST);

			if (__asprintf (&buf, _("connect to address %s: "),
					paddr) >= 0)
			  {
			    __fxprintf(NULL, "%s", buf);
			    free (buf);
			  }
			__set_errno (oerrno);
			perror(0);
			ai = ai->ai_next;
			getnameinfo(ai->ai_addr, ai->ai_addrlen,
				    paddr, sizeof(paddr),
				    NULL, 0,
				    NI_NUMERICHOST);
			if (__asprintf (&buf, _("Trying %s...\n"), paddr) >= 0)
			  {
			    __fxprintf (NULL, "%s", buf);
			    free (buf);
			  }
			continue;
		}
		if (refused && timo <= 16) {
			(void)__sleep(timo);
			timo *= 2;
			ai = res;
			refused = 0;
			continue;
		}
		freeaddrinfo(res);
		(void)__fxprintf(NULL, "%s: %s\n", *ahost,
				 __strerror_r(errno, errbuf, sizeof (errbuf)));
		__sigprocmask (SIG_SETMASK, &omask, 0);
		return -1;
	}
	lport--;
	if (fd2p == 0) {
		__write(s, "", 1);
		lport = 0;
	} else {
		char num[8];
		int s2 = rresvport_af(&lport, ai->ai_family), s3;
		socklen_t len = ai->ai_addrlen;

		if (s2 < 0)
			goto bad;
		__listen(s2, 1);
		(void)__snprintf(num, sizeof(num), "%d", lport);
		if (__write(s, num, strlen(num)+1) != (ssize_t)strlen(num)+1) {
			char *buf = NULL;

			if (__asprintf (&buf, _("\
rcmd: write (setting up stderr): %m\n")) >= 0)
			  {
			    __fxprintf(NULL, "%s", buf);
			    free (buf);
			  }
			(void)__close(s2);
			goto bad;
		}
		pfd[0].fd = s;
		pfd[1].fd = s2;
		__set_errno (0);
		if (__poll (pfd, 2, -1) < 1 || (pfd[1].revents & POLLIN) == 0){
			char *buf = NULL;

			if ((errno != 0
			     && __asprintf(&buf, _("\
rcmd: poll (setting up stderr): %m\n")) >= 0)
			    || (errno == 0
				&& __asprintf(&buf, _("\
poll: protocol failure in circuit setup\n")) >= 0))
			  {
			    __fxprintf (NULL, "%s", buf);
			    free  (buf);
			  }
			(void)__close(s2);
			goto bad;
		}
		s3 = TEMP_FAILURE_RETRY (accept(s2, &from.sa, &len));
		switch (from.sa.sa_family) {
		case AF_INET:
			rport = ntohs(from.sin.sin_port);
			break;
		case AF_INET6:
			rport = ntohs(from.sin6.sin6_port);
			break;
		default:
			rport = 0;
			break;
		}
		(void)__close(s2);
		if (s3 < 0) {
			(void)__fxprintf(NULL, "rcmd: accept: %m\n");
			lport = 0;
			goto bad;
		}
		*fd2p = s3;

		if (rport >= IPPORT_RESERVED || rport < IPPORT_RESERVED / 2){
			char *buf = NULL;

			if (__asprintf(&buf, _("\
socket: protocol failure in circuit setup\n")) >= 0)
			  {
			    __fxprintf (NULL, "%s", buf);
			    free (buf);
			  }
			goto bad2;
		}
	}
	struct iovec iov[3] =
	  {
	    [0] = { .iov_base = (void *) locuser,
		    .iov_len = strlen (locuser) + 1 },
	    [1] = { .iov_base = (void *) remuser,
		    .iov_len = strlen (remuser) + 1 },
	    [2] = { .iov_base = (void *) cmd,
		    .iov_len = strlen (cmd) + 1 }
	  };
	(void) TEMP_FAILURE_RETRY (__writev (s, iov, 3));
	n = TEMP_FAILURE_RETRY (__read(s, &c, 1));
	if (n != 1) {
		char *buf = NULL;

		if ((n == 0
		     && __asprintf(&buf, _("rcmd: %s: short read"),
				   *ahost) >= 0)
		    || (n != 0
			&& __asprintf(&buf, "rcmd: %s: %m\n", *ahost) >= 0))
		  {
		    __fxprintf (NULL, "%s", buf);
		    free (buf);
		  }
		goto bad2;
	}
	if (c != 0) {
		while (__read(s, &c, 1) == 1) {
			(void)__write(STDERR_FILENO, &c, 1);
			if (c == '\n')
				break;
		}
		goto bad2;
	}
	__sigprocmask (SIG_SETMASK, &omask, 0);
	freeaddrinfo(res);
	return s;
bad2:
	if (lport)
		(void)__close(*fd2p);
bad:
	(void)__close(s);
	__sigprocmask (SIG_SETMASK, &omask, 0);
	freeaddrinfo(res);
	return -1;
}
libc_hidden_def (rcmd_af)

int
rcmd (char **ahost, u_short rport, const char *locuser, const char *remuser,
      const char *cmd, int *fd2p)
{
  return rcmd_af (ahost, rport, locuser, remuser, cmd, fd2p, AF_INET);
}

int
rresvport_af (int *alport, sa_family_t family)
{
	union {
		struct sockaddr generic;
		struct sockaddr_in in;
		struct sockaddr_in6 in6;
	} ss;
	int s;
	size_t len;
	uint16_t *sport;

	switch(family){
	case AF_INET:
		len = sizeof(struct sockaddr_in);
		sport = &ss.in.sin_port;
		break;
	case AF_INET6:
		len = sizeof(struct sockaddr_in6);
		sport = &ss.in6.sin6_port;
		break;
	default:
		__set_errno (EAFNOSUPPORT);
		return -1;
	}
	/* NB: No SOCK_CLOEXEC for backwards compatibility.  */
	s = __socket(family, SOCK_STREAM, 0);
	if (s < 0)
		return -1;

	memset (&ss, '\0', sizeof(ss));
#ifdef SALEN
	ss.generic.__ss_len = len;
#endif
	ss.generic.sa_family = family;

	/* Ignore invalid values.  */
	if (*alport < IPPORT_RESERVED / 2)
		*alport = IPPORT_RESERVED / 2;
	else if (*alport >= IPPORT_RESERVED)
		*alport = IPPORT_RESERVED - 1;

	int start = *alport;
	do {
		*sport = htons((uint16_t) *alport);
		if (__bind(s, &ss.generic, len) >= 0)
			return s;
		if (errno != EADDRINUSE) {
			(void)__close(s);
			return -1;
		}
		if ((*alport)-- == IPPORT_RESERVED/2)
			*alport = IPPORT_RESERVED - 1;
	} while (*alport != start);
	(void)__close(s);
	__set_errno (EAGAIN);
	return -1;
}
libc_hidden_def (rresvport_af)

int
rresvport (int *alport)
{
	return rresvport_af(alport, AF_INET);
}

int	__check_rhosts_file = 1;
char	*__rcmd_errstr;

int
ruserok_af (const char *rhost, int superuser, const char *ruser,
	    const char *luser, sa_family_t af)
{
	struct addrinfo hints, *res, *res0;
	int gai;
	int ret;

	memset (&hints, '\0', sizeof(hints));
	hints.ai_family = af;
	gai = getaddrinfo(rhost, NULL, &hints, &res0);
	if (gai)
		return -1;
	ret = -1;
	for (res=res0; res; res=res->ai_next)
		if (ruserok2_sa(res->ai_addr, res->ai_addrlen,
				superuser, ruser, luser, rhost) == 0){
			ret = 0;
			break;
		}
	freeaddrinfo(res0);
	return (ret);
}
libc_hidden_def (ruserok_af)

int
ruserok (const char *rhost, int superuser, const char *ruser,
	 const char *luser)
{
	return ruserok_af(rhost, superuser, ruser, luser, AF_INET);
}

/* Extremely paranoid file open function. */
static FILE *
iruserfopen (const char *file, uid_t okuser)
{
  struct __stat64_t64 st;
  char *cp = NULL;
  FILE *res = NULL;

  /* If not a regular file, if owned by someone other than user or
     root, if writeable by anyone but the owner, or if hardlinked
     anywhere, quit.  */
  if (__lstat64_time64 (file, &st))
    cp = _("lstat failed");
  else if (!S_ISREG (st.st_mode))
    cp = _("not regular file");
  else
    {
      res = fopen (file, "rce");
      if (!res)
	cp = _("cannot open");
      else if (__fstat64_time64 (fileno (res), &st) < 0)
	cp = _("fstat failed");
      else if (st.st_uid && st.st_uid != okuser)
	cp = _("bad owner");
      else if (st.st_mode & (S_IWGRP|S_IWOTH))
	cp = _("writeable by other than owner");
      else if (st.st_nlink > 1)
	cp = _("hard linked somewhere");
    }

  /* If there were any problems, quit.  */
  if (cp != NULL)
    {
      __rcmd_errstr = cp;
      if (res)
	fclose (res);
      return NULL;
    }

  /* No threads use this stream.  */
  __fsetlocking (res, FSETLOCKING_BYCALLER);

  return res;
}

/*
 * New .rhosts strategy: We are passed an ip address. We spin through
 * hosts.equiv and .rhosts looking for a match. When the .rhosts only
 * has ip addresses, we don't have to trust a nameserver.  When it
 * contains hostnames, we spin through the list of addresses the nameserver
 * gives us and look for a match.
 *
 * Returns 0 if ok, -1 if not ok.
 */
static int
ruserok2_sa (struct sockaddr *ra, size_t ralen, int superuser,
	     const char *ruser, const char *luser, const char *rhost)
{
  FILE *hostf = NULL;
  int isbad = -1;

  if (!superuser)
    hostf = iruserfopen (_PATH_HEQUIV, 0);

  if (hostf)
    {
      isbad = __validuser2_sa (hostf, ra, ralen, luser, ruser, rhost);
      fclose (hostf);

      if (!isbad)
	return 0;
    }

  if (__check_rhosts_file || superuser)
    {
      char *pbuf;
      struct passwd pwdbuf, *pwd;
      size_t dirlen;
      size_t buflen = __sysconf (_SC_GETPW_R_SIZE_MAX);
      char *buffer = __alloca (buflen);
      uid_t uid;

      if (__getpwnam_r (luser, &pwdbuf, buffer, buflen, &pwd) != 0
	  || pwd == NULL)
	return -1;

      dirlen = strlen (pwd->pw_dir);
      pbuf = alloca (dirlen + sizeof "/.rhosts");
      __mempcpy (__mempcpy (pbuf, pwd->pw_dir, dirlen),
		 "/.rhosts", sizeof "/.rhosts");

       /* Change effective uid while reading .rhosts.  If root and
	  reading an NFS mounted file system, can't read files that
	  are protected read/write owner only.  */
       uid = __geteuid ();
       seteuid (pwd->pw_uid);
       hostf = iruserfopen (pbuf, pwd->pw_uid);

       if (hostf != NULL)
	 {
	   isbad = __validuser2_sa (hostf, ra, ralen, luser, ruser, rhost);
	   fclose (hostf);
	 }

       seteuid (uid);
       return isbad;
    }
  return -1;
}
/*
 * ruserok_sa() is now discussed on ipng, so
 * currently disabled for external use
 */
static int
ruserok_sa (struct sockaddr *ra, size_t ralen, int superuser,
	    const char *ruser, const char *luser)
{
  return ruserok2_sa(ra, ralen, superuser, ruser, luser, "-");
}

/* This is the exported version.  */
int
iruserok_af (const void *raddr, int superuser, const char *ruser,
	     const char *luser, sa_family_t af)
{
  union {
    struct sockaddr generic;
    struct sockaddr_in in;
    struct sockaddr_in6 in6;
  } ra;
  size_t ralen;

  memset (&ra, '\0', sizeof(ra));
  switch (af){
  case AF_INET:
    ra.in.sin_family = AF_INET;
    memcpy (&ra.in.sin_addr, raddr, sizeof(struct in_addr));
    ralen = sizeof(struct sockaddr_in);
    break;
  case AF_INET6:
    ra.in6.sin6_family = AF_INET6;
    memcpy (&ra.in6.sin6_addr, raddr, sizeof(struct in6_addr));
    ralen = sizeof(struct sockaddr_in6);
    break;
  default:
    return 0;
  }
  return ruserok_sa (&ra.generic, ralen, superuser, ruser, luser);
}
libc_hidden_def (iruserok_af)

int
iruserok (uint32_t raddr, int superuser, const char *ruser, const char *luser)
{
  return iruserok_af (&raddr, superuser, ruser, luser, AF_INET);
}

/*
 * XXX
 * Don't make static, used by lpd(8).
 *
 * This function is not used anymore. It is only present because lpd(8)
 * calls it (!?!). We simply call __invaliduser2() with an illegal rhost
 * argument. This means that netgroups won't work in .rhost/hosts.equiv
 * files. If you want lpd to work with netgroups, fix lpd to use ruserok()
 * or PAM.
 * Returns 0 if ok, -1 if not ok.
 */
int
__ivaliduser (FILE *hostf, uint32_t raddr, const char *luser,
	      const char *ruser)
{
	struct sockaddr_in ra;
	memset(&ra, '\0', sizeof(ra));
	ra.sin_family = AF_INET;
	ra.sin_addr.s_addr = raddr;
	return __validuser2_sa(hostf, (struct sockaddr *)&ra, sizeof(ra),
			       luser, ruser, "-");
}


/* Returns 1 on positive match, 0 on no match, -1 on negative match.  */
static int
__checkhost_sa (struct sockaddr *ra, size_t ralen, char *lhost,
		const char *rhost)
{
	struct addrinfo hints, *res0, *res;
	char raddr[INET6_ADDRSTRLEN];
	int match;
	int negate=1;    /* Multiply return with this to get -1 instead of 1 */

	/* Check nis netgroup.  */
	if (strncmp ("+@", lhost, 2) == 0)
		return innetgr (&lhost[2], rhost, NULL, NULL);

	if (strncmp ("-@", lhost, 2) == 0)
		return -innetgr (&lhost[2], rhost, NULL, NULL);

	/* -host */
	if (strncmp ("-", lhost,1) == 0) {
		negate = -1;
		lhost++;
	} else if (strcmp ("+",lhost) == 0) {
		return 1;                    /* asking for trouble, but ok.. */
	}

	/* Try for raw ip address first. */
	/* XXX */
	if (getnameinfo(ra, ralen,
			raddr, sizeof(raddr), NULL, 0,
			NI_NUMERICHOST) == 0
	    && strcmp(raddr, lhost) == 0)
		return negate;

	/* Better be a hostname. */
	match = 0;
	memset(&hints, '\0', sizeof(hints));
	hints.ai_family = ra->sa_family;
	if (getaddrinfo(lhost, NULL, &hints, &res0) == 0){
		/* Spin through ip addresses. */
		for (res = res0; res; res = res->ai_next)
		  {
		    if (res->ai_family == ra->sa_family
			&& !memcmp(res->ai_addr, ra, res->ai_addrlen))
		      {
			match = 1;
			break;
		      }
		  }
		freeaddrinfo (res0);
	}
	return negate * match;
}

/* Returns 1 on positive match, 0 on no match, -1 on negative match.  */
static int
__icheckuser (const char *luser, const char *ruser)
{
    /*
      luser is user entry from .rhosts/hosts.equiv file
      ruser is user id on remote host
      */

    /* [-+]@netgroup */
    if (strncmp ("+@", luser, 2) == 0)
	return innetgr (&luser[2], NULL, ruser, NULL);

    if (strncmp ("-@", luser,2) == 0)
	return -innetgr (&luser[2], NULL, ruser, NULL);

    /* -user */
    if (strncmp ("-", luser, 1) == 0)
	return -(strcmp (&luser[1], ruser) == 0);

    /* + */
    if (strcmp ("+", luser) == 0)
	return 1;

    /* simple string match */
    return strcmp (ruser, luser) == 0;
}

/*
 * Returns 1 for blank lines (or only comment lines) and 0 otherwise
 */
static int
__isempty (char *p)
{
    while (*p && isspace (*p)) {
	++p;
    }

    return (*p == '\0' || *p == '#') ? 1 : 0 ;
}

/*
 * Returns 0 if positive match, -1 if _not_ ok.
 */
static int
__validuser2_sa (FILE *hostf, struct sockaddr *ra, size_t ralen,
		 const char *luser, const char *ruser, const char *rhost)
{
    const char *user;
    char *p;
    int hcheck, ucheck;
    char *buf = NULL;
    size_t bufsize = 0;
    int retval = -1;

    while (__getline (&buf, &bufsize, hostf) > 0) {
	buf[bufsize - 1] = '\0'; /* Make sure it's terminated.  */
	p = buf;

	/* Skip empty or comment lines */
	if (__isempty (p)) {
	    continue;
	}

	for (;*p && !isspace(*p); ++p) {
	    *p = _tolower (*p);
	}

	/* Next we want to find the permitted name for the remote user.  */
	if (*p == ' ' || *p == '\t') {
	    /* <nul> terminate hostname and skip spaces */
	    for (*p++='\0'; *p && isspace (*p); ++p);

	    user = p;                   /* this is the user's name */
	    while (*p && !isspace (*p))
		++p;                    /* find end of user's name */
	} else
	    user = p;

	*p = '\0';              /* <nul> terminate username (+host?) */

	/* buf -> host(?) ; user -> username(?) */
	if (*buf == '\0')
	  break;
	if (*user == '\0')
	  user = luser;

	/* First check the user part.  In a naive implementation we
	   would check the host part first, then the user.  However,
	   if we check the user first and reject the entry we will
	   have saved doing any host lookups to normalize the comparison
	   and that likely saves several DNS queries.  Therefore we
	   check the user first.  */
	ucheck = __icheckuser (user, ruser);

	/* Either we found the user, or we didn't and this is a
	   negative host check.  We must do the negative host lookup
	   in order to preserve the semantics of stopping on this line
	   before processing others.  */
	if (ucheck != 0 || *buf == '-') {

	    /* Next check host part.  */
	    hcheck = __checkhost_sa (ra, ralen, buf, rhost);

	    /* Negative '-host user(?)' match?  */
	    if (hcheck < 0)
		break;

	    /* Positive 'host user' match?  */
	    if (hcheck > 0 && ucheck > 0) {
		retval = 0;
		break;
	    }

	    /* Negative 'host -user' match?  */
	    if (hcheck > 0 && ucheck < 0)
	      break;

	    /* Neither, go on looking for match.  */
	}
    }

    free (buf);

    return retval;
}
