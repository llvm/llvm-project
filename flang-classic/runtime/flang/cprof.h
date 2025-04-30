/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#define VERSION 0 /* file format version */

/* profiling data for lines and routines */

struct cinfo {
  double count; /* execution count */
  double cost;  /* cost in seconds */
  double time;  /* time used in seconds */
  double datas; /* number of data vectors sent */
  double bytes; /* number of bytes sent */
  double datar; /* number of data vectors received */
  double byter; /* number of bytes received */
  double datac; /* number of data vectors copied */
  double bytec; /* number of bytes copied */
};

/* function entry */

struct cprof {
  char *func;         /* function name */
  int funcl;          /* length of above */
  char *file;         /* file name */
  int filel;          /* length of above */
  int line;           /* beginning line number of routine */
  int lines;          /* number of lines in routine */
  struct cinfo d;     /* routine info */
  struct cinfo *cptr; /* pointer to vector of lines info */
};
