/* Argp example #2 -- a pretty minimal program using argp
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

/* This program doesn't use any options or arguments, but uses
   argp to be compliant with the GNU standard command line
   format.

   In addition to making sure no arguments are given, and
   implementing a --help option, this example will have a
   --version option, and will put the given documentation string
   and bug address in the --help output, as per GNU standards.

   The variable ARGP contains the argument parser specification;
   adding fields to this structure is the way most parameters are
   passed to argp_parse (the first three fields are usually used,
   but not in this small program).  There are also two global
   variables that argp knows about defined here,
   ARGP_PROGRAM_VERSION and ARGP_PROGRAM_BUG_ADDRESS (they are
   global variables because they will almost always be constant
   for a given program, even if it uses different argument
   parsers for various tasks).  */

#include <stdlib.h>
#include <argp.h>

const char *argp_program_version =
  "argp-ex2 1.0";
const char *argp_program_bug_address =
  "<bug-gnu-utils@@gnu.org>";

/* Program documentation.  */
static char doc[] =
  "Argp example #2 -- a pretty minimal program using argp";

/* Our argument parser.  The @code{options}, @code{parser}, and
   @code{args_doc} fields are zero because we have neither options or
   arguments; @code{doc} and @code{argp_program_bug_address} will be
   used in the output for @samp{--help}, and the @samp{--version}
   option will print out @code{argp_program_version}.  */
static struct argp argp = { 0, 0, 0, doc };

int
main (int argc, char **argv)
{
  argp_parse (&argp, argc, argv, 0, 0, 0);
  exit (0);
}
