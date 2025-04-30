/* Derived from the test case in
   https://sourceware.org/bugzilla/show_bug.cgi?id=714  */
#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>


static struct
{
  const char *chp;
  size_t nchp;
  const char *loc;
} tests[] =
{
  { (const char[]) { 0x8F, 0xA2, 0xAF }, 3, "ja_JP.EUC-JP" },
  { (const char[]) { 0xD1, 0xA5 }, 2, "ja_JP.EUC-JP" },
  { (const char[]) { 0x8E, 0xA5 }, 2, "ja_JP.EUC-JP" },
  { (const char[]) { 0x8E, 0xA2, 0xA1, 0xA1 }, 4, "zh_TW.EUC-TW" },
  { (const char[]) { 0xA1, 0xA1 }, 2, "zh_TW.EUC-TW" },
  { (const char[]) { 0xE3, 0x80, 0x80 }, 3, "de_DE.UTF-8" },
  { (const char[]) { 0xC3, 0xA4 }, 2, "de_DE.UTF-8" }
};
#define ntests (sizeof (tests) / sizeof (tests[0]))


static int t (const char *ch, size_t nch, const char *loc);

static int
do_test (void)
{
  int r = 0;
  for (int i = 0; i < ntests; ++i)
    r |= t (tests[i].chp, tests[i].nchp, tests[i].loc);
  return r;
}

static int
t (const char *ch, size_t nch, const char *loc)
{
  int i;
  wchar_t wch;
  wchar_t wch2;
  mbstate_t mbs;
  int n = 0;

  setlocale (LC_ALL, loc);

  memset (&mbs, '\0', sizeof (mbstate_t));
  for (i = 0; i < nch; i++)
    {
      n = mbrtowc (&wch, ch + i, 1, &mbs);
      if (n >= 0)
	break;
    }
  printf ("n = %d, count = %d, wch = %08lX\n", n, i, (unsigned long int) wch);

  memset (&mbs, '\0', sizeof (mbstate_t));
  n = mbrtowc (&wch2, ch, nch, &mbs);
  printf ("n = %d, wch = %08lX\n", n, (unsigned long int) wch2);

  int ret = n != nch || i + 1 != nch || n != nch || wch != wch2;
  puts (ret ? "FAIL\n" : "OK\n");
  return ret;
}

#include <support/test-driver.c>
