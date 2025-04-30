/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#include <argp.h>




#define OPT_TO_THREAD		300
#define OPT_TO_PROCESS		301
#define OPT_SYNC_SIGNAL		302
#define OPT_SYNC_JOIN		303
#define OPT_TOPLEVEL		304


static const struct argp_option test_options[] =
  {
    { NULL, 0, NULL, 0, "\
This is a test for threads so we allow ther user to selection the number of \
threads which are used at any one time.  Independently the total number of \
rounds can be selected.  This is the total number of threads which will have \
run when the process terminates:" },
    { "threads", 't', "NUMBER", 0, "Number of threads used at once" },
    { "starts", 's', "NUMBER", 0, "Total number of working threads" },
    { "toplevel", OPT_TOPLEVEL, "NUMBER", 0,
      "Number of toplevel threads which start the other threads; this \
implies --sync-join" },

    { NULL, 0, NULL, 0, "\
Each thread can do one of two things: sleep or do work.  The latter is 100% \
CPU bound.  The work load is the probability a thread does work.  All values \
from zero to 100 (inclusive) are valid.  How often each thread repeats this \
can be determined by the number of rounds.  The work cost determines how long \
each work session (not sleeping) takes.  If it is zero a thread would \
effectively nothing.  By setting the number of rounds to zero the thread \
does no work at all and pure thread creation times can be measured." },
    { "workload", 'w', "PERCENT", 0, "Percentage of time spent working" },
    { "workcost", 'c', "NUMBER", 0,
      "Factor in the cost of each round of working" },
    { "rounds", 'r', "NUMBER", 0, "Number of rounds each thread runs" },

    { NULL, 0, NULL, 0, "\
There are a number of different methods how thread creation can be \
synchronized.  Synchronization is necessary since the number of concurrently \
running threads is limited." },
    { "sync-signal", OPT_SYNC_SIGNAL, NULL, 0,
      "Synchronize using a signal (default)" },
    { "sync-join", OPT_SYNC_JOIN, NULL, 0, "Synchronize using pthread_join" },

    { NULL, 0, NULL, 0, "\
One parameter for each threads execution is the size of the stack.  If this \
parameter is not used the system's default stack size is used.  If many \
threads are used the stack size should be chosen quite small." },
    { "stacksize", 'S', "BYTES", 0, "Size of threads stack" },
    { "guardsize", 'g', "BYTES", 0,
      "Size of stack guard area; must fit into the stack" },

    { NULL, 0, NULL, 0, "Signal options:" },
    { "to-thread", OPT_TO_THREAD, NULL, 0, "Send signal to main thread" },
    { "to-process", OPT_TO_PROCESS, NULL, 0,
      "Send signal to process (default)" },

    { NULL, 0, NULL, 0, "Administrative options:" },
    { "progress", 'p', NULL, 0, "Show signs of progress" },
    { "timing", 'T', NULL, 0,
      "Measure time from startup to the last thread finishing" },
    { NULL, 0, NULL, 0, NULL }
  };

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  test_options, parse_opt
};


static int
do_test (void)
{
  int argc = 2;
  char *argv[3] = { (char *) "tst-argp1", (char *) "--help", NULL };
  int remaining;

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  return 0;
}


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  return ARGP_ERR_UNKNOWN;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
