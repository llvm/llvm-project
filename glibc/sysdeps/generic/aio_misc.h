/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#ifndef _AIO_MISC_H
#define _AIO_MISC_H	1

#include <aio.h>
#include <pthread.h>


/* Extend the operation enum.  */
enum
{
  LIO_DSYNC = LIO_NOP + 1,
  LIO_SYNC,
  LIO_READ64 = LIO_READ | 128,
  LIO_WRITE64 = LIO_WRITE | 128
};


/* Union of the two request types.  */
typedef union
  {
    struct aiocb aiocb;
    struct aiocb64 aiocb64;
  } aiocb_union;


/* Used to synchronize.  */
struct waitlist
  {
    struct waitlist *next;

    /* The next two fields is used in synchronous `lio_listio' operations.  */
#ifndef DONT_NEED_AIO_MISC_COND
    pthread_cond_t *cond;
#endif
    int *result;

    volatile unsigned int *counterp;
    /* The next field is used in asynchronous `lio_listio' operations.  */
    struct sigevent *sigevp;
  };


/* Status of a request.  */
enum
{
  no,
  queued,
  yes,
  allocated,
  done
};


/* Used to queue requests..  */
struct requestlist
  {
    int running;

    struct requestlist *last_fd;
    struct requestlist *next_fd;
    struct requestlist *next_prio;
    struct requestlist *next_run;

    /* Pointer to the actual data.  */
    aiocb_union *aiocbp;

    /* List of waiting processes.  */
    struct waitlist *waiting;
  };


/* Lock for global I/O list of requests.  */
extern pthread_mutex_t __aio_requests_mutex attribute_hidden;


/* Enqueue request.  */
extern struct requestlist *__aio_enqueue_request (aiocb_union *aiocbp,
						  int operation)
  attribute_hidden;

/* Find request entry for given AIO control block.  */
extern struct requestlist *__aio_find_req (aiocb_union *elem) attribute_hidden;

/* Find request entry for given file descriptor.  */
extern struct requestlist *__aio_find_req_fd (int fildes) attribute_hidden;

/* Remove request from the list.  */
extern void __aio_remove_request (struct requestlist *last,
				  struct requestlist *req, int all)
  attribute_hidden;

/* Release the entry for the request.  */
extern void __aio_free_request (struct requestlist *req) attribute_hidden;

/* Notify initiator of request and tell this everybody listening.  */
extern void __aio_notify (struct requestlist *req) attribute_hidden;

/* Notify initiator of request.  */
extern int __aio_notify_only (struct sigevent *sigev) attribute_hidden;

/* Send the signal.  */
extern int __aio_sigqueue (int sig, const union sigval val, pid_t caller_pid)
  attribute_hidden;

#endif /* aio_misc.h */
