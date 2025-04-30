/* Open File Description Locks Usage Example
   Copyright (C) 1991-2021 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>

#define FILENAME	"/tmp/foo"
#define NUM_THREADS	3
#define ITERATIONS	5

void *
thread_start (void *arg)
{
  int i, fd, len;
  long tid = (long) arg;
  char buf[256];
  struct flock lck = {
    .l_whence = SEEK_SET,
    .l_start = 0,
    .l_len = 1,
  };

  fd = open ("/tmp/foo", O_RDWR | O_CREAT, 0666);

  for (i = 0; i < ITERATIONS; i++)
    {
      lck.l_type = F_WRLCK;
      fcntl (fd, F_OFD_SETLKW, &lck);

      len = sprintf (buf, "%d: tid=%ld fd=%d\n", i, tid, fd);

      lseek (fd, 0, SEEK_END);
      write (fd, buf, len);
      fsync (fd);

      lck.l_type = F_UNLCK;
      fcntl (fd, F_OFD_SETLK, &lck);

      /* sleep to ensure lock is yielded to another thread */
      usleep (1);
    }
  pthread_exit (NULL);
}

int
main (int argc, char **argv)
{
  long i;
  pthread_t threads[NUM_THREADS];

  truncate (FILENAME, 0);

  for (i = 0; i < NUM_THREADS; i++)
    pthread_create (&threads[i], NULL, thread_start, (void *) i);

  pthread_exit (NULL);
  return 0;
}
