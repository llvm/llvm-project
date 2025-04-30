/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * The external routines in this module are:
 *
 * Fio_asy_open - called from open
 * Fio_asy_enable - enable async i/o, disable stdio
 * Fio_asy_read - async read
 * Fio_asy_write - async write
 * Fio_asy_start - for vectored i/o, start reads or writes
 * Fio_asy_disable - disable async i/o, enable stdio
 * Fio_asy_close - called from close
 */

#if !defined(TARGET_WIN)
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <aio.h>
#include <signal.h>
#else
#include <windows.h>
#include <errno.h>
#endif

#include "stdioInterf.h"
#include "async.h"


#define FIO_MAX_ASYNC_TRANSACTIONS 16

/* one struct per file */

struct asy_transaction_data {
  long len;
  seekoffx_t off;
};

#if defined(TARGET_WIN)
struct asy {
  FILE *fp;
  int fd;
  HANDLE handle;
  int flags;
  int outstanding_transactions;
  struct asy_transaction_data atd[FIO_MAX_ASYNC_TRANSACTIONS];
  OVERLAPPED overlap[FIO_MAX_ASYNC_TRANSACTIONS];
};

union Converter {
  struct {
    DWORD wOffset;
    DWORD wOffsetHigh;
  };
  seekoffx_t offset;
};

#else

struct asy {
  FILE *fp;
  int fd;
  int flags;
  int outstanding_transactions;
  struct asy_transaction_data atd[FIO_MAX_ASYNC_TRANSACTIONS];
  struct aiocb aiocb[FIO_MAX_ASYNC_TRANSACTIONS];
};
#endif

/* flags */

#define ASY_FDACT 0x01 /* fd is active, not fp */
#define ASY_IOACT 0x02 /* asynch i/o is active */

static int slime;

/* internal wait for asynch i/o */

#if defined(TARGET_WIN)
static int
asy_wait(struct asy *asy)
{
  long len;
  int s;
  int tn;
  BOOL aio_complete;

  if (!(asy->flags & ASY_IOACT)) {
    return (0);
  }
  asy->flags &= ~ASY_IOACT;

  for (tn = 0; tn < asy->outstanding_transactions; tn++) {
    do {
      aio_complete =
          GetOverlappedResult(asy->handle, &(asy->overlap[tn]), &len, FALSE);
    } while ((aio_complete == FALSE));

    if (slime)
      printf("---Fio_asy_wait %d\n", asy->fd);

    if (asy->atd[tn].len != len) {
      __io_set_errno(FIO_EEOF);
      printf("Wait FIO_EEOF error forthcoming.\n");
      return (-1);
    }
  }
  asy->atd[0].off = asy->atd[asy->outstanding_transactions].off;
  asy->outstanding_transactions = 0;
  return (0);
}
#else
static int
asy_wait(struct asy *asy)
{
  long len;
  int s;
  int tn;
  struct aiocb *p[1];

  if (!(asy->flags & ASY_IOACT)) { /* i/o active? */
    return (0);
  }
  asy->flags &= ~ASY_IOACT;

  for (tn = 0; tn < asy->outstanding_transactions; tn++) {
    p[0] = &(asy->aiocb[tn]);
    do {
      s = aio_suspend((const struct aiocb * const *)p, 1,
                      (const struct timespec *)0);
    } while ((s == -1) && (__io_errno() == EINTR));
    if (s == -1) {
      return (-1);
    }
    if (slime)
      printf("---Fio_asy_wait %d\n", asy->fd);
    len = aio_return(&(asy->aiocb[tn]));
    if (len == -1) {
      s = aio_error(&(asy->aiocb[tn]));
      __io_set_errno(s);
      return (-1);
    }
    if (asy->atd[tn].len != len) { /* incomplete transfer? */
      __io_set_errno(FIO_EEOF);  /* ..yes */
      return (-1);
    }
  }
  /* Reset the number of outstanding transactions back to 0,
   * and set the offset to the end of the last transaction list
   */
  asy->atd[0].off = asy->atd[asy->outstanding_transactions].off;
  asy->outstanding_transactions = 0;
  return (0);
}
#endif

int
Fio_asy_fseek(struct asy *asy, long offset, int whence)
{
  if (slime)
    printf("--Fio_asy_seek %d %ld\n", asy->fd, offset);

  if (whence == SEEK_CUR) {
    asy->atd[asy->outstanding_transactions].off += offset;
  } else {
    asy->atd[asy->outstanding_transactions].off = offset;
  }
  return (0);
}

/* enable fd, disable fp */

int
Fio_asy_enable(struct asy *asy)
{
  int n;

  if (slime)
    printf("--Fio_asy_enable %d\n", asy->fd);
  if (asy->flags & ASY_IOACT) { /* i/o active? */
    if (asy_wait(asy) == -1) {
      return (-1);
    }
  }
  if (asy->flags & ASY_FDACT) { /* fd already active? */
    return (0);
  }

  asy->atd[0].off = __io_ftellx(asy->fp);
  asy->outstanding_transactions = 0;
  if (asy->atd[0].off == -1) {
    return (-1);
  }
  n = __io_fflush(asy->fp);
  if (n != 0) {
    return (-1);
  }
  asy->flags |= ASY_FDACT; /* fd is now active */
  return (0);
}

/* disable fd, enable fp */

int
Fio_asy_disable(struct asy *asy)
{
  int n;
  int offset;

  if (slime)
    printf("--Fio_asy_disable %d\n", asy->fd);
  if (asy->flags & ASY_IOACT) { /* i/o active? */
    if (asy_wait(asy) == -1) {
      return (-1);
    }
  }
  if (!(asy->flags & ASY_FDACT)) { /* fd not active? */
    return (0);
  }
  /* Seek to the end of the the list. */
  offset = asy->atd[asy->outstanding_transactions].off;
  n = __io_fseekx(asy->fp, offset, 0);
  if (n == -1) {
    return (-1);
  }
  asy->flags &= ~ASY_FDACT; /* fd is now inactive */
  return (0);
}

/* init file for asynch i/o, called from open */

int
Fio_asy_open(FILE *fp, struct asy **pasy)
{
  struct asy *asy;
#if defined(TARGET_WIN)
  HANDLE temp_handle;
#endif
  asy = (struct asy *)calloc(sizeof(struct asy), 1);
  if (asy == (struct asy *)0) {
    __io_set_errno(ENOMEM);
    return (-1);
  }
  asy->fp = fp;
  asy->fd = __io_getfd(fp);
#if defined(TARGET_WIN)
  temp_handle = _get_osfhandle(asy->fd);
  asy->handle =
      ReOpenFile(temp_handle, GENERIC_READ | GENERIC_WRITE,
                 FILE_SHARE_READ | FILE_SHARE_WRITE, FILE_FLAG_OVERLAPPED);
  if (asy->handle == INVALID_HANDLE_VALUE) {
    __io_set_errno(EBADF);
    return (-1);
  }
#endif
  if (slime)
    printf("--Fio_asy_open %d\n", asy->fd);
  *pasy = asy;
  return (0);
}

/* start an asynch read */

int
Fio_asy_read(struct asy *asy, void *adr, long len)
{
  int n;
  int tn;

#if defined(TARGET_WIN)
  union Converter converter;
#endif
  if (slime)
    printf("--Fio_asy_read %d %p %ld\n", asy->fd, adr, len);

#if defined(TARGET_WIN)
  if (asy->flags & ASY_IOACT) { /* i/o active? */
    if (asy_wait(asy) == -1) {  /* ..yes, wait */
      return (-1);
    }
  }
  tn = asy->outstanding_transactions;
  asy->overlap[tn].Internal = 0;
  asy->overlap[tn].InternalHigh = 0;
  asy->overlap[tn].Pointer = 0;
  /* Load asy->off into OffsetHigh/Offset */
  converter.offset = asy->atd[tn].off;
  asy->overlap[tn].Offset = converter.wOffset;
  asy->overlap[tn].OffsetHigh = converter.wOffsetHigh;
  asy->overlap[tn].hEvent = 0;
  if (ReadFile(asy->handle, adr, len, NULL, &(asy->overlap[tn])) == FALSE &&
      GetLastError() != ERROR_IO_PENDING) {
    n = -1;
  }
#else
  tn = asy->outstanding_transactions;
  asy->aiocb[tn].aio_fildes = asy->fd;
  asy->aiocb[tn].aio_reqprio = 0;
  asy->aiocb[tn].aio_buf = adr;
  asy->aiocb[tn].aio_nbytes = len;
  memset(&(asy->aiocb[tn].aio_sigevent), 0, sizeof(struct sigevent));
  asy->aiocb[tn].aio_offset = asy->atd[tn].off;
  n = aio_read(&(asy->aiocb[tn]));
#endif

  if (n == -1) {
    return (-1);
  }
  asy->atd[tn].len = len;
  asy->atd[tn + 1].off = asy->atd[tn].off + len;
  asy->flags |= ASY_IOACT; /* i/o now active */
  asy->outstanding_transactions += 1;
  return (0);
}

/* start an asynch write */

int
Fio_asy_write(struct asy *asy, void *adr, long len)
{
  int n;
  int tn;
#if defined(TARGET_WIN)
  union Converter converter;
#endif

  if (slime)
    printf("--Fio_asy_write %d %p %ld\n", asy->fd, adr, len);

#if defined(TARGET_WIN)
  if (asy->flags & ASY_IOACT) { /* i/o active? */
    if (asy_wait(asy) == -1) {  /* ..yes, wait */
      return (-1);
    }
  }
  tn = asy->outstanding_transactions;
  asy->overlap[tn].Internal = 0;
  asy->overlap[tn].InternalHigh = 0;
  asy->overlap[tn].Pointer = 0;
  /* Load asy->off into OffsetHigh/Offset. */
  converter.offset = asy->atd[0].off;
  asy->overlap[tn].Offset = converter.wOffset;
  asy->overlap[tn].OffsetHigh = converter.wOffsetHigh;
  asy->overlap[tn].hEvent = 0;
  if (WriteFile(asy->handle, adr, len, NULL, &(asy->overlap[tn])) == FALSE &&
      GetLastError() != ERROR_IO_PENDING) {
    n = -1;
  }
#else
  tn = asy->outstanding_transactions;
  asy->aiocb[tn].aio_fildes = asy->fd;
  asy->aiocb[tn].aio_reqprio = 0;
  asy->aiocb[tn].aio_buf = adr;
  asy->aiocb[tn].aio_nbytes = len;
  memset(&(asy->aiocb[tn].aio_sigevent), 0, sizeof(struct sigevent));
  asy->aiocb[tn].aio_offset = asy->atd[tn].off;
  n = aio_write(&(asy->aiocb[tn]));
#endif

  if (n == -1) {
    return (-1);
  }
  asy->atd[tn].len = len;
  asy->atd[tn + 1].off = asy->atd[tn].off + len;
  asy->outstanding_transactions += 1;
  asy->flags |= ASY_IOACT; /* i/o now active */
  return (0);
}

int
Fio_asy_start(struct asy *asy)
{
  if (slime)
    printf("--Fio_asy_start %d\n", asy->fd);
  return (0);
}

/* close asynch i/o called from close */

int
Fio_asy_close(struct asy *asy)
{
  int n;

  if (slime)
    printf("--Fio_asy_close %d\n", asy->fd);
  n = 0;
  if (asy->flags & ASY_IOACT) { /* i/o active? */
    n = asy_wait(asy);
  }
#if defined(TARGET_WIN)
  /* Close the Re-opened handle that we created. */
  CloseHandle(asy->handle);
#endif
  free(asy);
  return (n);
}

