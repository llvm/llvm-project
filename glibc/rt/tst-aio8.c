#include <aio.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int
do_test (void)
{
  int fd = open ("/dev/full", O_RDWR);
  if (fd == -1)
    {
      puts ("could not open /dev/full");
      return 0;
    }

  struct aiocb req;
  req.aio_fildes = fd;
  req.aio_lio_opcode = LIO_WRITE;
  req.aio_reqprio = 0;
  req.aio_buf = (void *) "hello";
  req.aio_nbytes = 5;
  req.aio_offset = 0;
  req.aio_sigevent.sigev_notify = SIGEV_NONE;

  struct aiocb *list[1];
  list[0] = &req;

  int r = lio_listio (LIO_WAIT, list, 1, NULL);
  int e = errno;

  printf ("r = %d, e = %d (%s)\n", r, e, strerror (e));

  return r != -1 || e != EIO;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
