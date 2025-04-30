/* Byte Stream Connection Server Example
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

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define PORT	5555
#define MAXMSG	512

int
read_from_client (int filedes)
{
  char buffer[MAXMSG];
  int nbytes;

  nbytes = read (filedes, buffer, MAXMSG);
  if (nbytes < 0)
    {
      /* Read error. */
      perror ("read");
      exit (EXIT_FAILURE);
    }
  else if (nbytes == 0)
    /* End-of-file. */
    return -1;
  else
    {
      /* Data read. */
      fprintf (stderr, "Server: got message: `%s'\n", buffer);
      return 0;
    }
}

int
main (void)
{
  extern int make_socket (uint16_t port);
  int sock;
  fd_set active_fd_set, read_fd_set;
  int i;
  struct sockaddr_in clientname;
  size_t size;

  /* Create the socket and set it up to accept connections. */
  sock = make_socket (PORT);
  if (listen (sock, 1) < 0)
    {
      perror ("listen");
      exit (EXIT_FAILURE);
    }

  /* Initialize the set of active sockets. */
  FD_ZERO (&active_fd_set);
  FD_SET (sock, &active_fd_set);

  while (1)
    {
      /* Block until input arrives on one or more active sockets. */
      read_fd_set = active_fd_set;
      if (select (FD_SETSIZE, &read_fd_set, NULL, NULL, NULL) < 0)
	{
	  perror ("select");
	  exit (EXIT_FAILURE);
	}

      /* Service all the sockets with input pending. */
      for (i = 0; i < FD_SETSIZE; ++i)
	if (FD_ISSET (i, &read_fd_set))
	  {
	    if (i == sock)
	      {
		/* Connection request on original socket. */
		int new;
		size = sizeof (clientname);
		new = accept (sock,
			      (struct sockaddr *) &clientname,
			      &size);
		if (new < 0)
		  {
		    perror ("accept");
		    exit (EXIT_FAILURE);
		  }
		fprintf (stderr,
			 "Server: connect from host %s, port %hd.\n",
			 inet_ntoa (clientname.sin_addr),
			 ntohs (clientname.sin_port));
		FD_SET (new, &active_fd_set);
	      }
	    else
	      {
		/* Data arriving on an already-connected socket. */
		if (read_from_client (i) < 0)
		  {
		    close (i);
		    FD_CLR (i, &active_fd_set);
		  }
	      }
	  }
    }
}
