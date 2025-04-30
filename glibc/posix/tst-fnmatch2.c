#include <fnmatch.h>
#include <stdio.h>

int
do_test (void)
{
  char pattern[] = "a*b*c*d*e*f*g*h*i*j*k*l*m*n*o*p*q*r*s*t*u*v*w*x*y*z*";
  const char *string = "aaaabbbbccccddddeeeeffffgggghhhhiiiijjjjkkkkllllmmmm"
		       "nnnnooooppppqqqqrrrrssssttttuuuuvvvvwwwwxxxxyyyy";
  if (fnmatch (pattern, string, 0) != FNM_NOMATCH)
    {
      puts ("First fnmatch didn't return FNM_NOMATCH");
      return 1;
    }
  pattern[(sizeof pattern) - 3] = '*';
  if (fnmatch (pattern, string, 0) != 0)
    {
      puts ("Second fnmatch didn't return 0");
      return 1;
    }
  if (fnmatch ("a*b/*", "abbb/.x", FNM_PATHNAME | FNM_PERIOD) != FNM_NOMATCH)
    {
      puts ("Third fnmatch didn't return FNM_NOMATCH");
      return 1;
    }
  if (fnmatch ("a*b/*", "abbb/xy", FNM_PATHNAME | FNM_PERIOD) != 0)
    {
      puts ("Fourth fnmatch didn't return 0");
      return 1;
    }
  if (fnmatch ("[", "[", 0) != 0)
    {
      puts ("Fifth fnmatch didn't return 0");
      return 1;
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
