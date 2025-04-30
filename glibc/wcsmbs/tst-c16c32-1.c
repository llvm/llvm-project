#include <inttypes.h>
#include <locale.h>
#include <stdio.h>
#include <uchar.h>
#include <stdint.h>

static int
do_test (void)
{
  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    {
      puts ("cannot set locale");
      return 1;
    }

  int result = 0;

  char32_t c32 = 48;
  do
    {
      if (c32 >= 0xd800 && c32 <= 0xe000)
	continue;

      char buf[20];
      size_t n1 = c32rtomb (buf, c32, NULL);
      if (n1 <= 0)
	{
	  printf ("c32rtomb for U'\\x%" PRIx32 "' failed\n", (uint32_t) c32);
	  result = 1;
	  continue;
	}

      char32_t c32out;
      size_t n2 = mbrtoc32 (&c32out, buf, n1, NULL);
      if ((ssize_t) n2 < 0)
	{
	  printf ("mbrtoc32 for U'\\x%" PRIx32 "' failed\n", (uint32_t) c32);
	  result = 1;
	  continue;
	}
      if (n2 != n1)
	{
	  printf ("mbrtoc32 for U'\\x%" PRIx32 "' consumed %zu bytes, not %zu\n",
		  (uint32_t) c32, n2, n1);
	  result = 1;
	}
      else if (c32out != c32)
	{
	  printf ("mbrtoc32 for U'\\x%" PRIx32 "' produced U'\\x%" PRIx32 "\n",
		  (uint32_t) c32, (uint32_t) c32out);
	  result = 1;
	}

      char16_t c16;
      size_t n3 = mbrtoc16 (&c16, buf, n1, NULL);
      if (n3 != n1)
	{
	  printf ("mbrtoc16 for U'\\x%" PRIx32 "' did not consume all bytes\n",
		  (uint32_t) c32);
	  result = 1;
	  continue;
	}
      if (c32 < 0x10000)
	{
	  if (c16 != c32)
	    {
	      printf ("mbrtoc16 for U'\\x%" PRIx32 "' produce u'\\x%" PRIx16 "'\n",
		      (uint32_t) c32, (uint16_t) c16);
	      result = 1;
	      continue;
	    }
	}
      else
	{
	  buf[0] = '1';
	  char16_t c16_2;
	  size_t n4 = mbrtoc16 (&c16_2, buf, 1, NULL);
	  if (n4 != (size_t) -3)
	    {
	      printf ("second mbrtoc16 for U'\\x%" PRIx32 "' did not return -3\n",
		      (uint32_t) c32);
	      result = 1;
	      continue;
	    }

	  if (c32 != (((uint32_t) (c16 - 0xd7c0)) << 10) + (c16_2 - 0xdc00))
	    {
	      printf ("mbrtoc16 for U'\\x%" PRIx32 "' returns U'\\x%" PRIx32 "\n",
		      (uint32_t) c32,
		      (((uint32_t) (c16 - 0xd7c0)) << 10) + (c16_2 - 0xdc00));
	      result = 1;
	      continue;
	    }
	}

      buf[0] = '\0';
      char16_t c16_nul;
      n3 = mbrtoc16 (&c16_nul, buf, n1, NULL);
      if (n3 != 0)
	{
	  printf ("mbrtoc16 for '\\0' returns %zd\n", n3);
	  result = 1;
	  continue;
	}

      if (c32 < 0x10000)
	{
	  size_t n5 = c16rtomb (buf, c16, NULL);
	  if ((ssize_t) n5 < 0)
	    {
	      printf ("c16rtomb for U'\\x%" PRIx32 "' failed with %zd\n",
		      (uint32_t) c32, n5);
	      result = 1;
	      continue;
	    }
	  if (n5 != n1)
	    {
	      printf ("c16rtomb for U'\\x%" PRIx32 "' produced %zu bytes instead of %zu bytes\n",
		      (uint32_t) c32, n5, n1);
	      result = 1;
	      continue;
	    }
	}
    }
  while ((c32 += 0x1111) <= U'\x12000');

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
