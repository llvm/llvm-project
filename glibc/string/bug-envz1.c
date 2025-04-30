/* Test for bug BZ #2703.  */
#include <stdio.h>
#include <envz.h>
#include <stdlib.h>
#include <string.h>

static const struct
{
  const char *s;
  int in_result;
} strs[] =
{
  { "a=1", 1 },
  { "b=2", 1 },
  { "(*)", 0 },
  { "(*)", 0 },
  { "e=5", 1 },
  { "f=", 1 },
  { "(*)", 0 },
  { "h=8", 1 },
  { "i=9", 1 },
  { "j", 0 }
};

#define nstrs (sizeof (strs) / sizeof (strs[0]))


int
do_test (void)
{

  size_t size = 0;
  char *str = malloc (100);
  if (str == NULL)
    {
      puts ("out of memory");
      return 1;
    }

  char **argz = &str;

  for (int i = 0; i < nstrs; ++i)
    argz_add_sep (argz, &size, strs[i].s, '\0');

  printf ("calling envz_strip with size=%zu\n", size);
  envz_strip (argz, &size);

  int result = 0;
  printf ("new size=%zu\n", size);
  for (int i = 0; i < nstrs; ++i)
    if (strs[i].in_result)
      {
        char name[2];
        name[0] = strs[i].s[0];
        name[1] = '\0';

        char *e = envz_entry (*argz, size, name);
        if (e == NULL)
          {
            printf ("entry '%s' not found\n", name);
            result = 1;
          }
        else if (strcmp (e, strs[i].s) != 0)
          {
            printf ("entry '%s' does not match: is '%s', expected '%s'\n",
                    name, e, strs[i].s);
            result = 1;
          }
      }

  free (*argz);
  return result;
}

#include <support/test-driver.c>
