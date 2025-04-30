/*
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

/*
 * Open two pipes to a child process, one for reading, one for writing.
 * The pipes are accessed by FILE pointers. This is NOT a public
 * interface, but for internal use only!
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <rpc/rpc.h>
#include <rpc/clnt.h>

#include <libio/iolibio.h>
#define fflush(s) _IO_fflush (s)
#define __fdopen(fd,m) _IO_fdopen (fd,m)

/*
 * returns pid, or -1 for failure
 */
int
_openchild (const char *command, FILE ** fto, FILE ** ffrom)
{
  int i;
  int pid;
  int pdto[2];
  int pdfrom[2];

  if (__pipe (pdto) < 0)
    goto error1;
  if (__pipe (pdfrom) < 0)
    goto error2;
  switch (pid = __fork ())
    {
    case -1:
      goto error3;

    case 0:
      /*
       * child: read from pdto[0], write into pdfrom[1]
       */
      __close (0);
      __dup (pdto[0]);
      __close (1);
      __dup (pdfrom[1]);
      fflush (stderr);
      for (i = _rpc_dtablesize () - 1; i >= 3; i--)
	__close (i);
      fflush (stderr);
      execlp (command, command, NULL);
      perror ("exec");
      _exit (~0);

    default:
      /*
       * parent: write into pdto[1], read from pdfrom[0]
       */
      *fto = __fdopen (pdto[1], "w");
      __close (pdto[0]);
      *ffrom = __fdopen (pdfrom[0], "r");
      __close (pdfrom[1]);
      break;
    }
  return pid;

  /*
   * error cleanup and return
   */
error3:
  __close (pdfrom[0]);
  __close (pdfrom[1]);
error2:
  __close (pdto[0]);
  __close (pdto[1]);
error1:
  return -1;
}
