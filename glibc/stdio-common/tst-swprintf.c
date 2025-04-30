#include <array_length.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

/* This is the relevant piece from the charmap:
<UFF61>     /x8e/xa1     HALFWIDTH IDEOGRAPHIC FULL STOP
<UFF62>     /x8e/xa2     HALFWIDTH LEFT CORNER BRACKET
<UFF63>     /x8e/xa3     HALFWIDTH RIGHT CORNER BRACKET
<UFF64>     /x8e/xa4     HALFWIDTH IDEOGRAPHIC COMMA
 */

const char input[] = "\x8e\xa1g\x8e\xa2h\x8e\xa3i\x8e\xa4j";

static int
do_test (void)
{
  wchar_t buf[1000];
  int result = 0;
  ssize_t n;

  if (setlocale (LC_ALL, "ja_JP.EUC-JP") == NULL)
    {
      puts ("cannot set locale");
      exit (1);
    }

#define CHECK(fmt, nexp, exp) \
  n = swprintf (buf, array_length (buf), fmt, input);			      \
  if (n != nexp)							      \
    {									      \
      printf ("swprintf (.., .., L\"%ls\", \"%ls\") return %d, not %d\n",     \
	      fmt, (wchar_t*) input, (int) n, (int) nexp);		      \
      result = 1;							      \
    }									      \
  else if (wcscmp (buf, exp) != 0)					      \
    {									      \
      printf ("\
swprintf (.., .., L\"%ls\", \"%ls\") produced \"%ls\", not \"%ls\"\n",	      \
	     fmt, (wchar_t *) input, buf, exp );			      \
      result = 1;							      \
    }

  CHECK (L"[%-6.0s]", 8, L"[      ]");
  CHECK (L"[%-6.1s]", 8, L"[\xff61     ]");
  CHECK (L"[%-6.2s]", 8, L"[\xff61g    ]");
  CHECK (L"[%-6.3s]", 8, L"[\xff61g\xff62   ]");
  CHECK (L"[%-6.4s]", 8, L"[\xff61g\xff62h  ]");
  CHECK (L"[%-6.5s]", 8, L"[\xff61g\xff62h\xff63 ]");
  CHECK (L"[%-6.6s]", 8, L"[\xff61g\xff62h\xff63i]");
  CHECK (L"[%-6.7s]", 9, L"[\xff61g\xff62h\xff63i\xff64]");
  CHECK (L"[%-6.8s]", 10, L"[\xff61g\xff62h\xff63i\xff64j]");

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
