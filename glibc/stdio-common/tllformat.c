#include <stdio.h>
#include <string.h>

/* The original file was tiformat.c and it has been changed for long long tests\
. */
typedef struct
{
  int line;
  long long int value;
  const char *result;
  const char *format_string;
} sprint_int_type;

sprint_int_type sprint_ints[] =
{
  {__LINE__, 0x00000000ULL,             "0", "%llx"},
  {__LINE__, 0xffff00000000208bULL,     "ffff00000000208b", "%llx"},
  {__LINE__, 0xffff00000000208bULL,     "18446462598732849291", "%llu"},
  {__LINE__, 18446462598732849291ULL,   "ffff00000000208b", "%llx"},
  {__LINE__, 18446462598732849291ULL,   "18446462598732849291", "%llu"},
  {__LINE__, 18359476226655002763ULL,   "fec9f65b0000208b", "%llx"},
  {__LINE__, 18359476226655002763ULL,   "18359476226655002763", "%llu"},

  {0},
};

int
main (void)
{
  int errcount = 0;
  int testcount = 0;
#define BSIZE 1024
  char buffer[BSIZE];
  sprint_int_type *iptr;
  for (iptr = sprint_ints; iptr->line; iptr++)
    {
      sprintf (buffer, iptr->format_string, iptr->value);
      if (strcmp (buffer, iptr->result) != 0)
	{
	  ++errcount;
	  printf ("\
Error in line %d using \"%s\".  Result is \"%s\"; should be: \"%s\".\n",
		  iptr->line, iptr->format_string, buffer, iptr->result);
        }
      ++testcount;
    }

  if (errcount == 0)
    {
      printf ("Encountered no errors in %d tests.\n", testcount);
      return 0;
    }
  else
    {
      printf ("Encountered %d errors in %d tests.\n",
	      errcount, testcount);
      return 1;
    }
}
