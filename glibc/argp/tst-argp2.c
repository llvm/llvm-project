/* Copyright (C) 2007-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2007.

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

static const struct argp_option opt1[] =
  {
    { "opt1", '1', "NUMBER", 0, "Option 1" },
    { NULL, 0, NULL, 0, NULL }
  };

static const struct argp_option opt2[] =
  {
    { "opt2", '2', "NUMBER", 0, "Option 2" },
    { NULL, 0, NULL, 0, NULL }
  };

static const struct argp_option opt3[] =
  {
    { "opt3", '3', "NUMBER", 0, "Option 3" },
    { NULL, 0, NULL, 0, NULL }
  };

static const struct argp_option opt4[] =
  {
    { "opt4", '4', "NUMBER", 0, "Option 4" },
    { NULL, 0, NULL, 0, NULL }
  };

static const struct argp_option opt5[] =
  {
    { "opt5", '5', "NUMBER", 0, "Option 5" },
    { NULL, 0, NULL, 0, NULL }
  };

static struct argp argp5 =
  {
    opt5, NULL, "args doc5", "doc5", NULL, NULL, NULL
  };

static struct argp argp4 =
  {
    opt4, NULL, "args doc4", "doc4", NULL, NULL, NULL
  };

static struct argp argp3 =
  {
    opt3, NULL, "args doc3", "doc3", NULL, NULL, NULL
  };

static struct argp_child children2[] =
  {
    { &argp4, 0, "child3", 3 },
    { &argp5, 0, "child4", 4 },
    { NULL, 0, NULL, 0 }
  };

static struct argp argp2 =
  {
    opt2, NULL, "args doc2", "doc2", children2, NULL, NULL
  };

static struct argp_child children1[] =
  {
    { &argp2, 0, "child1", 1 },
    { &argp3, 0, "child2", 2 },
    { NULL, 0, NULL, 0 }
  };

static struct argp argp1 =
  {
    opt1, NULL, "args doc1", "doc1", children1, NULL, NULL
  };


static int
do_test (void)
{
  argp_help (&argp1, stdout, ARGP_HELP_LONG, (char *) "tst-argp2");
  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
