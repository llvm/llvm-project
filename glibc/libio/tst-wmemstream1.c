#include <wchar.h>

extern int fclose (FILE*);

#if defined __GNUC__ && __GNUC__ >= 11
/* Verify that calling fclose on the result of open_wmemstream doesn't
   trigger GCC -Wmismatched-dealloc with fclose forward-declared and
   without <stdio.h> included first (it is included later, in.
   "tst-memstream1.c").  */
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wmismatched-dealloc"
#endif

int test_open_wmemstream_no_stdio (void)
{
  {
    wchar_t *buf;
    size_t size;
    FILE *f = open_wmemstream (&buf, &size);
    fclose (f);
  }

  {
    FILE* (*pf)(wchar_t**, size_t*) = open_wmemstream;
    wchar_t *buf;
    size_t size;
    FILE *f = pf (&buf, &size);
    fclose (f);
  }
  return 0;
}

#if defined __GNUC__ && __GNUC__ >= 11
/* Restore -Wmismatched-dealloc setting.  */
# pragma GCC diagnostic pop
#endif

#define CHAR_T wchar_t
#define W(o) L##o
#define OPEN_MEMSTREAM open_wmemstream

#include "tst-memstream1.c"
