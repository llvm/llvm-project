/* Test program for argp argument parser
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Miles Bader <miles@gnu.ai.mit.edu>.

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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <argp.h>

const char *argp_program_version = "argp-test 1.0";

struct argp_option sub_options[] =
{
  {"subopt1",       's',     0,  0, "Nested option 1"},
  {"subopt2",       'S',     0,  0, "Nested option 2"},

  { 0, 0, 0, 0, "Some more nested options:", 10},
  {"subopt3",       'p',     0,  0, "Nested option 3"},

  {"subopt4",       'q',     0,  0, "Nested option 4", 1},

  {0}
};

static const char sub_args_doc[] = "STRING...\n-";
static const char sub_doc[] = "\vThis is the doc string from the sub-arg-parser.";

static error_t
sub_parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARGP_KEY_NO_ARGS:
      printf ("NO SUB ARGS\n");
      break;
    case ARGP_KEY_ARG:
      printf ("SUB ARG: %s\n", arg);
      break;

    case 's' : case 'S': case 'p': case 'q':
      printf ("SUB KEY %c\n", key);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static char *
sub_help_filter (int key, const char *text, void *input)
{
  if (key == ARGP_KEY_HELP_EXTRA)
    return strdup ("This is some extra text from the sub parser (note that it \
is preceded by a blank line).");
  else
    return (char *)text;
}

static struct argp sub_argp = {
  sub_options, sub_parse_opt, sub_args_doc, sub_doc, 0, sub_help_filter
};

/* Structure used to communicate with the parsing functions.  */
struct params
{
  unsigned foonly;		/* Value parsed for foonly.  */
  unsigned foonly_default;	/* Default value for it.  */
};

#define OPT_PGRP 1
#define OPT_SESS 2

struct argp_option options[] =
{
  {"pid",       'p',     "PID", 0, "List the process PID"},
  {"pgrp",      OPT_PGRP,"PGRP",0, "List processes in the process group PGRP"},
  {"no-parent", 'P',	 0,     0, "Include processes without parents"},
  {0,           'x',     0,     OPTION_ALIAS},
  {"all-fields",'Q',     0,     0, "Don't elide unusable fields (normally"
				   " if there's some reason ps can't"
				   " print a field for any process, it's"
				   " removed from the output entirely)" },
  {"reverse",   'r',    0,      0, "Reverse the order of any sort"},
  {"gratuitously-long-reverse-option", 0, 0, OPTION_ALIAS},
  {"session",  OPT_SESS,"SID",  OPTION_ARG_OPTIONAL,
				   "Add the processes from the session"
				   " SID (which defaults to the sid of"
				   " the current process)" },

  {0,0,0,0, "Here are some more options:"},
  {"foonly", 'f', "ZOT", OPTION_ARG_OPTIONAL, "Glork a foonly"},
  {"zaza", 'z', 0, 0, "Snit a zar"},

  {0}
};

static const char args_doc[] = "STRING";
static const char doc[] = "Test program for argp."
 "\vThis doc string comes after the options."
 "\nHey!  Some manual formatting!"
 "\nThe current time is: %s";

static void
popt (int key, char *arg)
{
  char buf[12];
  if (isprint (key))
    sprintf (buf, "%c", key);
  else
    sprintf (buf, "%d", key);
  if (arg)
    printf ("KEY %s: %s\n", buf, arg);
  else
    printf ("KEY %s\n", buf);
}

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  struct params *params = state->input;

  switch (key)
    {
    case ARGP_KEY_NO_ARGS:
      printf ("NO ARGS\n");
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num > 0)
	return ARGP_ERR_UNKNOWN; /* Leave it for the sub-arg parser.  */
      printf ("ARG: %s\n", arg);
      break;

    case 'f':
      if (arg)
	params->foonly = atoi (arg);
      else
	params->foonly = params->foonly_default;
      popt (key, arg);
      break;

    case 'p': case 'P': case OPT_PGRP: case 'x': case 'Q':
    case 'r': case OPT_SESS: case 'z':
      popt (key, arg);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static char *
help_filter (int key, const char *text, void *input)
{
  char *new_text;
  struct params *params = input;

  if (key == ARGP_KEY_HELP_POST_DOC && text)
    {
      time_t now = time (0);
      asprintf (&new_text, text, ctime (&now));
    }
  else if (key == 'f')
    /* Show the default for the --foonly option.  */
    asprintf (&new_text, "%s (ZOT defaults to %x)",
	      text, params->foonly_default);
  else
    new_text = (char *)text;

  return new_text;
}

static struct argp_child argp_children[] = { { &sub_argp }, { 0 } };
static struct argp argp = {
  options, parse_opt, args_doc, doc, argp_children, help_filter
};

int
main (int argc, char **argv)
{
  struct params params;
  params.foonly = 0;
  params.foonly_default = random ();
  argp_parse (&argp, argc, argv, 0, 0, &params);
  printf ("After parsing: foonly = %x\n", params.foonly);
  return 0;
}
