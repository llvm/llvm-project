/*
 * UFC-crypt: ultra fast crypt(3) implementation
 *
 * Copyright (C) 1991-2021 Free Software Foundation, Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with the GNU C Library; see the file COPYING.LIB.  If not,
 * see <https://www.gnu.org/licenses/>.
 *
 * @(#)ufc.c	2.7 9/10/96
 *
 * Stub main program for debugging
 * and benchmarking.
 *
 */

#include <stdio.h>

char *crypt();

main(argc, argv)
  int argc;
  char **argv;
  { char *s;
    unsigned long i,iterations;

    if(argc != 2) {
      fprintf(stderr, "usage: ufc iterations\n");
      exit(1);
    }
    argv++;
    iterations = atoi(*argv);
    printf("ufc: running %d iterations\n", iterations);

    for(i=0; i<iterations; i++)
      s=crypt("foob","ar");
    if(strcmp(s, "arlEKn0OzVJn.") == 0)
      printf("OK\n");
    else {
      printf("wrong result: %s!!\n", s);
      exit(1);
    }
    exit(0);
  }
