#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

int
main (void)
{
  char *sp1, *sp2, *sp3, *sp4;
  wchar_t *lsp1, *lsp2, *lsp3, *lsp4;
  int result = 0;
  char buf[2048+64];
  size_t i;

#define FAIL() \
  do {							\
    result = 1;						\
    printf ("test at line %d failed\n", __LINE__);	\
  } while (0)

  setlocale (LC_ALL, "de_DE.UTF-8");
  if (sscanf ("A  \xc3\x84-\t\t\xc3\x84-abcdefbcd\t\xc3\x84-B",
	      "A%ms%10ms%4m[bcd]%4mcB", &sp1, &sp2, &sp3, &sp4) != 4)
    FAIL ();
  else
    {
      if (strcmp (sp1, "\xc3\x84-") != 0)
	FAIL ();
      free (sp1);
      if (strcmp (sp2, "\xc3\x84-abcdefb") != 0)
	FAIL ();
      free (sp2);
      if (strcmp (sp3, "cd") != 0)
	FAIL ();
      free (sp3);
      if (memcmp (sp4, "\t\xc3\x84-", 4) != 0)
	FAIL ();
      free (sp4);
    }

  if (sscanf ("A  \xc3\x84-\t\t\xc3\x84-abcdefbcd\t\xc3\x84-BB",
	      "A%mS%10mls%4ml[bcd]%4mCB", &lsp1, &lsp2, &lsp3, &lsp4) != 4)
    FAIL ();
  else
    {
      if (wcscmp (lsp1, L"\xc4-") != 0)
	FAIL ();
      free (lsp1);
      if (wcscmp (lsp2, L"\xc4-abcdefbc") != 0)
	FAIL ();
      free (lsp2);
      if (wcscmp (lsp3, L"d") != 0)
	FAIL ();
      free (lsp3);
      if (memcmp (lsp4, L"\t\xc4-B", 4 * sizeof (wchar_t)) != 0)
	FAIL ();
      free (lsp4);
    }

  memset (buf, '/', sizeof (buf));
  buf[0] = '\t';
  buf[1] = ' ';
  buf[2] = 0xc3;
  buf[3] = 0x84;
  buf[2048] = 0xc3;
  buf[2049] = 0x84;
  buf[2058] = '\t';
  buf[2059] = 'a';
  if (sscanf (buf, "%ms%mc", &sp1, &sp2) != 2)
    FAIL ();
  else
    {
      if (sp1[0] != '\xc3' || sp1[1] != '\x84'
	  || sp1[2046] != '\xc3' || sp1[2047] != '\x84'
	  || sp1[2056] != '\0')
	FAIL ();
      sp1[2046] = '/';
      sp1[2047] = '/';
      for (i = 2; i < 2056; i++)
	if (sp1[i] != '/')
	  FAIL ();
      free (sp1);
      if (sp2[0] != '\t')
	FAIL ();
      free (sp2);
    }
  if (sscanf (buf, "%2048ms%mc", &sp3, &sp4) != 2)
    FAIL ();
  else
    {
      if (sp3[0] != '\xc3' || sp3[1] != '\x84'
	  || sp3[2046] != '\xc3' || sp3[2047] != '\x84'
	  || sp3[2048] != '\0')
	FAIL ();
      for (i = 2; i < 2046; i++)
	if (sp3[i] != '/')
	  FAIL ();
      free (sp3);
      if (sp4[0] != '/')
	FAIL ();
      free (sp4);
    }
  if (sscanf (buf, "%4mc%1500m[dr/]%548m[abc/d]%3mc", &sp1, &sp2, &sp3, &sp4)
      != 4)
    FAIL ();
  else
    {
      if (memcmp (sp1, "\t \xc3\x84", 4) != 0)
	FAIL ();
      free (sp1);
      for (i = 0; i < 1500; i++)
	if (sp2[i] != '/')
	  FAIL ();
      if (sp2[1500] != '\0')
	FAIL ();
      free (sp2);
      for (i = 0; i < 544; i++)
	if (sp3[i] != '/')
	  FAIL ();
      if (sp3[544] != '\0')
	FAIL ();
      free (sp3);
      if (memcmp (sp4, "\xc3\x84/", 3) != 0)
	FAIL ();
      free (sp4);
    }
  if (sscanf (buf, "%mS%mC", &lsp1, &lsp2) != 2)
    FAIL ();
  else
    {
      if (lsp1[0] != L'\xc4' || lsp1[2045] != L'\xc4'
	  || lsp1[2054] != L'\0')
	FAIL ();
      lsp1[2045] = L'/';
      for (i = 1; i < 2054; i++)
	if (lsp1[i] != L'/')
	  FAIL ();
      free (lsp1);
      if (lsp2[0] != L'\t')
	FAIL ();
      free (lsp2);
    }
  if (sscanf (buf, "%2048mls%mlc", &lsp3, &lsp4) != 2)
    FAIL ();
  else
    {
      if (lsp3[0] != L'\xc4' || lsp3[2045] != L'\xc4'
	  || lsp3[2048] != L'\0')
	FAIL ();
      lsp3[2045] = L'/';
      for (i = 1; i < 2048; i++)
	if (lsp3[i] != L'/')
	  FAIL ();
      free (lsp3);
      if (lsp4[0] != L'/')
	FAIL ();
      free (lsp4);
    }
  if (sscanf (buf, "%4mC%1500ml[dr/]%548ml[abc/d]%3mlc",
	      &lsp1, &lsp2, &lsp3, &lsp4) != 4)
    FAIL ();
  else
    {
      if (memcmp (lsp1, L"\t \xc4/", 4 * sizeof (wchar_t)) != 0)
	FAIL ();
      free (lsp1);
      for (i = 0; i < 1500; i++)
	if (lsp2[i] != L'/')
	  FAIL ();
      if (lsp2[1500] != L'\0')
	FAIL ();
      free (lsp2);
      for (i = 0; i < 543; i++)
	if (lsp3[i] != L'/')
	  FAIL ();
      if (lsp3[543] != L'\0')
	FAIL ();
      free (lsp3);
      if (memcmp (lsp4, L"\xc4//", 3 * sizeof (wchar_t)) != 0)
	FAIL ();
      free (lsp4);
    }

  return result;
}
