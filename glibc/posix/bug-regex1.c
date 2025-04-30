/* Test case by Jim Meyering <jim@meyering.net>.  */
#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <regex.h>
#include <wchar.h>

int
main (void)
{
  struct re_pattern_buffer regex;
  struct re_registers regs;
  const char *s;
  int match;
  int result = 0;

  memset (&regex, '\0', sizeof (regex));

  setlocale (LC_ALL, "de_DE.ISO-8859-1");
  fwide (stdout, -1);

  re_set_syntax (RE_SYNTAX_POSIX_EGREP | RE_DEBUG);

  puts ("in C locale");
  setlocale (LC_ALL, "C");
  s = re_compile_pattern ("[anù]*n", 7, &regex);
  if (s != NULL)
    {
      puts ("re_compile_pattern return non-NULL value");
      result = 1;
    }
  else
    {
      match = re_match (&regex, "an", 2, 0, &regs);
      if (match != 2)
	{
	  printf ("re_match returned %d, expected 2\n", match);
	  result = 1;
	}
      else
	puts (" -> OK");
    }

  puts ("in de_DE.ISO-8859-1 locale");
  setlocale (LC_ALL, "de_DE.ISO-8859-1");
  s = re_compile_pattern ("[anù]*n", 7, &regex);
  if (s != NULL)
    {
      puts ("re_compile_pattern return non-NULL value");
      result = 1;
    }
  else
    {
      match = re_match (&regex, "an", 2, 0, &regs);
      if (match != 2)
	{
	  printf ("re_match returned %d, expected 2\n", match);
	  result = 1;
	}
      else
	puts (" -> OK");
    }

  return result;
}
