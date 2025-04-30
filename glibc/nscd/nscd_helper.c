/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <not-cancel.h>
#include <kernel-features.h>
#include <nss.h>
#include <struct___timespec64.h>

#include "nscd-client.h"

/* Extra time we wait if the socket is still receiving data.  This
   value is in milliseconds.  Note that the other side is nscd on the
   local machine and it is already transmitting data.  So the wait
   time need not be long.  */
#define EXTRA_RECEIVE_TIME 200


static int
wait_on_socket (int sock, long int usectmo)
{
  struct pollfd fds[1];
  fds[0].fd = sock;
  fds[0].events = POLLIN | POLLERR | POLLHUP;
  int n = __poll (fds, 1, usectmo);
  if (n == -1 && __builtin_expect (errno == EINTR, 0))
    {
      /* Handle the case where the poll() call is interrupted by a
	 signal.  We cannot just use TEMP_FAILURE_RETRY since it might
	 lead to infinite loops.  */
      struct __timespec64 now;
      __clock_gettime64 (CLOCK_REALTIME, &now);
      int64_t end = (now.tv_sec * 1000 + usectmo
                     + (now.tv_nsec + 500000) / 1000000);
      long int timeout = usectmo;
      while (1)
	{
	  n = __poll (fds, 1, timeout);
	  if (n != -1 || errno != EINTR)
	    break;

	  /* Recompute the timeout time.  */
          __clock_gettime64 (CLOCK_REALTIME, &now);
	  timeout = end - ((now.tv_sec * 1000
                            + (now.tv_nsec + 500000) / 1000000));
	}
    }

  return n;
}


ssize_t
__readall (int fd, void *buf, size_t len)
{
  size_t n = len;
  ssize_t ret;
  do
    {
    again:
      ret = TEMP_FAILURE_RETRY (__read (fd, buf, n));
      if (ret <= 0)
	{
	  if (__builtin_expect (ret < 0 && errno == EAGAIN, 0)
	      /* The socket is still receiving data.  Wait a bit more.  */
	      && wait_on_socket (fd, EXTRA_RECEIVE_TIME) > 0)
	    goto again;

	  break;
	}
      buf = (char *) buf + ret;
      n -= ret;
    }
  while (n > 0);
  return ret < 0 ? ret : len - n;
}


ssize_t
__readvall (int fd, const struct iovec *iov, int iovcnt)
{
  ssize_t ret = TEMP_FAILURE_RETRY (__readv (fd, iov, iovcnt));
  if (ret <= 0)
    {
      if (__glibc_likely (ret == 0 || errno != EAGAIN))
	/* A genuine error or no data to read.  */
	return ret;

      /* The data has not all yet been received.  Do as if we have not
	 read anything yet.  */
      ret = 0;
    }

  size_t total = 0;
  for (int i = 0; i < iovcnt; ++i)
    total += iov[i].iov_len;

  if (ret < total)
    {
      struct iovec iov_buf[iovcnt];
      ssize_t r = ret;

      struct iovec *iovp = memcpy (iov_buf, iov, iovcnt * sizeof (*iov));
      do
	{
	  while (iovp->iov_len <= r)
	    {
	      r -= iovp->iov_len;
	      --iovcnt;
	      ++iovp;
	    }
	  iovp->iov_base = (char *) iovp->iov_base + r;
	  iovp->iov_len -= r;
	again:
	  r = TEMP_FAILURE_RETRY (__readv (fd, iovp, iovcnt));
	  if (r <= 0)
	    {
	      if (__builtin_expect (r < 0 && errno == EAGAIN, 0)
		  /* The socket is still receiving data.  Wait a bit more.  */
		  && wait_on_socket (fd, EXTRA_RECEIVE_TIME) > 0)
		goto again;

	      break;
	    }
	  ret += r;
	}
      while (ret < total);
      if (r < 0)
	ret = r;
    }
  return ret;
}


static int
open_socket (request_type type, const char *key, size_t keylen)
{
  int sock;

  sock = __socket (PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (sock < 0)
    return -1;

  size_t real_sizeof_reqdata = sizeof (request_header) + keylen;
  struct
  {
    request_header req;
    char key[];
  } *reqdata = alloca (real_sizeof_reqdata);

  struct sockaddr_un sun;
  sun.sun_family = AF_UNIX;
  strcpy (sun.sun_path, _PATH_NSCDSOCKET);
  if (__connect (sock, (struct sockaddr *) &sun, sizeof (sun)) < 0
      && errno != EINPROGRESS)
    goto out;

  reqdata->req.version = NSCD_VERSION;
  reqdata->req.type = type;
  reqdata->req.key_len = keylen;

  memcpy (reqdata->key, key, keylen);

  bool first_try = true;
  struct __timespec64 tvend = { 0, 0 };
  while (1)
    {
#ifndef MSG_NOSIGNAL
# define MSG_NOSIGNAL 0
#endif
      ssize_t wres = TEMP_FAILURE_RETRY (__send (sock, reqdata,
						 real_sizeof_reqdata,
						 MSG_NOSIGNAL));
      if (__glibc_likely (wres == (ssize_t) real_sizeof_reqdata))
	/* We managed to send the request.  */
	return sock;

      if (wres != -1 || errno != EAGAIN)
	/* Something is really wrong, no chance to continue.  */
	break;

      /* The daemon is busy wait for it.  */
      int to;
      struct __timespec64 now;
      __clock_gettime64 (CLOCK_REALTIME, &now);
      if (first_try)
	{
	  tvend.tv_nsec = now.tv_nsec;
	  tvend.tv_sec = now.tv_sec + 5;
	  to = 5 * 1000;
	  first_try = false;
	}
      else
	to = ((tvend.tv_sec - now.tv_sec) * 1000
	      + (tvend.tv_nsec - now.tv_nsec) / 1000000);

      struct pollfd fds[1];
      fds[0].fd = sock;
      fds[0].events = POLLOUT | POLLERR | POLLHUP;
      if (__poll (fds, 1, to) <= 0)
	/* The connection timed out or broke down.  */
	break;

      /* We try to write again.  */
    }

 out:
  __close_nocancel_nostatus (sock);

  return -1;
}


void
__nscd_unmap (struct mapped_database *mapped)
{
  assert (mapped->counter == 0);
  __munmap ((void *) mapped->head, mapped->mapsize);
  free (mapped);
}


/* Try to get a file descriptor for the shared meory segment
   containing the database.  */
struct mapped_database *
__nscd_get_mapping (request_type type, const char *key,
		    struct mapped_database **mappedp)
{
  struct mapped_database *result = NO_MAPPING;
#ifdef SCM_RIGHTS
  const size_t keylen = strlen (key) + 1;
  int saved_errno = errno;

  int mapfd = -1;
  char resdata[keylen];

  /* Open a socket and send the request.  */
  int sock = open_socket (type, key, keylen);
  if (sock < 0)
    goto out;

  /* Room for the data sent along with the file descriptor.  We expect
     the key name back.  */
  uint64_t mapsize;
  struct iovec iov[2];
  iov[0].iov_base = resdata;
  iov[0].iov_len = keylen;
  iov[1].iov_base = &mapsize;
  iov[1].iov_len = sizeof (mapsize);

  union
  {
    struct cmsghdr hdr;
    char bytes[CMSG_SPACE (sizeof (int))];
  } buf;
  struct msghdr msg = { .msg_iov = iov, .msg_iovlen = 2,
			.msg_control = buf.bytes,
			.msg_controllen = sizeof (buf) };
  struct cmsghdr *cmsg = CMSG_FIRSTHDR (&msg);

  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN (sizeof (int));

  /* This access is well-aligned since BUF is correctly aligned for an
     int and CMSG_DATA preserves this alignment.  */
  memset (CMSG_DATA (cmsg), '\xff', sizeof (int));

  msg.msg_controllen = cmsg->cmsg_len;

  if (wait_on_socket (sock, 5 * 1000) <= 0)
    goto out_close2;

# ifndef MSG_CMSG_CLOEXEC
#  define MSG_CMSG_CLOEXEC 0
# endif
  ssize_t n = TEMP_FAILURE_RETRY (__recvmsg (sock, &msg, MSG_CMSG_CLOEXEC));

  if (__builtin_expect (CMSG_FIRSTHDR (&msg) == NULL
			|| (CMSG_FIRSTHDR (&msg)->cmsg_len
			    != CMSG_LEN (sizeof (int))), 0))
    goto out_close2;

  int *ip = (void *) CMSG_DATA (cmsg);
  mapfd = *ip;

  if (__glibc_unlikely (n != keylen && n != keylen + sizeof (mapsize)))
    goto out_close;

  if (__glibc_unlikely (strcmp (resdata, key) != 0))
    goto out_close;

  if (__glibc_unlikely (n == keylen))
    {
      struct __stat64_t64 st;
      if (__glibc_unlikely (__fstat64_time64 (mapfd, &st) != 0)
	  || __builtin_expect (st.st_size < sizeof (struct database_pers_head),
			       0))
	goto out_close;

      mapsize = st.st_size;
    }

  /* The file is large enough, map it now.  */
  void *mapping = __mmap (NULL, mapsize, PROT_READ, MAP_SHARED, mapfd, 0);
  if (__glibc_likely (mapping != MAP_FAILED))
    {
      /* Check whether the database is correct and up-to-date.  */
      struct database_pers_head *head = mapping;

      if (__builtin_expect (head->version != DB_VERSION, 0)
	  || __builtin_expect (head->header_size != sizeof (*head), 0)
	  /* Catch some misconfiguration.  The server should catch
	     them now but some older versions did not.  */
	  || __builtin_expect (head->module == 0, 0)
	  /* This really should not happen but who knows, maybe the update
	     thread got stuck.  */
	  || __builtin_expect (! head->nscd_certainly_running
			       && (head->timestamp + MAPPING_TIMEOUT
				   < time_now ()), 0))
	{
	out_unmap:
	  __munmap (mapping, mapsize);
	  goto out_close;
	}

      size_t size = (sizeof (*head) + roundup (head->module * sizeof (ref_t),
					       ALIGN)
		     + head->data_size);

      if (__glibc_unlikely (mapsize < size))
	goto out_unmap;

      /* Allocate a record for the mapping.  */
      struct mapped_database *newp = malloc (sizeof (*newp));
      if (newp == NULL)
	/* Ugh, after all we went through the memory allocation failed.  */
	goto out_unmap;

      newp->head = mapping;
      newp->data = ((char *) mapping + head->header_size
		    + roundup (head->module * sizeof (ref_t), ALIGN));
      newp->mapsize = size;
      newp->datasize = head->data_size;
      /* Set counter to 1 to show it is usable.  */
      newp->counter = 1;

      result = newp;
    }

 out_close:
  __close (mapfd);
 out_close2:
  __close (sock);
 out:
  __set_errno (saved_errno);
#endif	/* SCM_RIGHTS */

  struct mapped_database *oldval = *mappedp;
  *mappedp = result;

  if (oldval != NULL && atomic_decrement_val (&oldval->counter) == 0)
    __nscd_unmap (oldval);

  return result;
}

struct mapped_database *
__nscd_get_map_ref (request_type type, const char *name,
		    volatile struct locked_map_ptr *mapptr, int *gc_cyclep)
{
  struct mapped_database *cur = mapptr->mapped;
  if (cur == NO_MAPPING)
    return cur;

  if (!__nscd_acquire_maplock (mapptr))
    return NO_MAPPING;

  cur = mapptr->mapped;

  if (__glibc_likely (cur != NO_MAPPING))
    {
      /* If not mapped or timestamp not updated, request new map.  */
      if (cur == NULL
	  || (cur->head->nscd_certainly_running == 0
	      && cur->head->timestamp + MAPPING_TIMEOUT < time_now ())
	  || cur->head->data_size > cur->datasize)
	cur = __nscd_get_mapping (type, name,
				  (struct mapped_database **) &mapptr->mapped);

      if (__glibc_likely (cur != NO_MAPPING))
	{
	  if (__builtin_expect (((*gc_cyclep = cur->head->gc_cycle) & 1) != 0,
				0))
	    cur = NO_MAPPING;
	  else
	    atomic_increment (&cur->counter);
	}
    }

  mapptr->lock = 0;

  return cur;
}


/* Using sizeof (hashentry) is not always correct to determine the size of
   the data structure as found in the nscd cache.  The program could be
   a 64-bit process and nscd could be a 32-bit process.  In this case
   sizeof (hashentry) would overestimate the size.  The following is
   the minimum size of such an entry, good enough for our tests here.  */
#define MINIMUM_HASHENTRY_SIZE \
  (offsetof (struct hashentry, dellist) + sizeof (int32_t))

/* Don't return const struct datahead *, as eventhough the record
   is normally constant, it can change arbitrarily during nscd
   garbage collection.  */
struct datahead *
__nscd_cache_search (request_type type, const char *key, size_t keylen,
		     const struct mapped_database *mapped, size_t datalen)
{
  unsigned long int hash = __nss_hash (key, keylen) % mapped->head->module;
  size_t datasize = mapped->datasize;

  ref_t trail = mapped->head->array[hash];
  trail = atomic_forced_read (trail);
  ref_t work = trail;
  size_t loop_cnt = datasize / (MINIMUM_HASHENTRY_SIZE
				+ offsetof (struct datahead, data) / 2);
  int tick = 0;

  while (work != ENDREF && work + MINIMUM_HASHENTRY_SIZE <= datasize)
    {
      struct hashentry *here = (struct hashentry *) (mapped->data + work);
      ref_t here_key, here_packet;

#if !_STRING_ARCH_unaligned
      /* Although during garbage collection when moving struct hashentry
	 records around we first copy from old to new location and then
	 adjust pointer from previous hashentry to it, there is no barrier
	 between those memory writes.  It is very unlikely to hit it,
	 so check alignment only if a misaligned load can crash the
	 application.  */
      if ((uintptr_t) here & (__alignof__ (*here) - 1))
	return NULL;
#endif

      if (type == here->type
	  && keylen == here->len
	  && (here_key = atomic_forced_read (here->key)) + keylen <= datasize
	  && memcmp (key, mapped->data + here_key, keylen) == 0
	  && ((here_packet = atomic_forced_read (here->packet))
	      + sizeof (struct datahead) <= datasize))
	{
	  /* We found the entry.  Increment the appropriate counter.  */
	  struct datahead *dh
	    = (struct datahead *) (mapped->data + here_packet);

#if !_STRING_ARCH_unaligned
	  if ((uintptr_t) dh & (__alignof__ (*dh) - 1))
	    return NULL;
#endif

	  /* See whether we must ignore the entry or whether something
	     is wrong because garbage collection is in progress.  */
	  if (dh->usable
	      && here_packet + dh->allocsize <= datasize
	      && (here_packet + offsetof (struct datahead, data) + datalen
		  <= datasize))
	    return dh;
	}

      work = atomic_forced_read (here->next);
      /* Prevent endless loops.  This should never happen but perhaps
	 the database got corrupted, accidentally or deliberately.  */
      if (work == trail || loop_cnt-- == 0)
	break;
      if (tick)
	{
	  struct hashentry *trailelem;
	  trailelem = (struct hashentry *) (mapped->data + trail);

#if !_STRING_ARCH_unaligned
	  /* We have to redo the checks.  Maybe the data changed.  */
	  if ((uintptr_t) trailelem & (__alignof__ (*trailelem) - 1))
	    return NULL;
#endif

	  if (trail + MINIMUM_HASHENTRY_SIZE > datasize)
	    return NULL;

	  trail = atomic_forced_read (trailelem->next);
	}
      tick = 1 - tick;
    }

  return NULL;
}


/* Create a socket connected to a name. */
int
__nscd_open_socket (const char *key, size_t keylen, request_type type,
		    void *response, size_t responselen)
{
  /* This should never happen and it is something the nscd daemon
     enforces, too.  He it helps to limit the amount of stack
     used.  */
  if (keylen > MAXKEYLEN)
    return -1;

  int saved_errno = errno;

  int sock = open_socket (type, key, keylen);
  if (sock >= 0)
    {
      /* Wait for data.  */
      if (wait_on_socket (sock, 5 * 1000) > 0)
	{
	  ssize_t nbytes = TEMP_FAILURE_RETRY (__read (sock, response,
						       responselen));
	  if (nbytes == (ssize_t) responselen)
	    return sock;
	}

      __close_nocancel_nostatus (sock);
    }

  __set_errno (saved_errno);

  return -1;
}
