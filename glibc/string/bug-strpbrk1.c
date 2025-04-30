/* Test case by Joseph S. Myers <jsm28@cam.ac.uk>.  */
#undef __USE_STRING_INLINES
#define __USE_STRING_INLINES
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <libc-diag.h>

int
main (void)
{
  const char *a = "abc";
  const char *b = a;

  DIAG_PUSH_NEEDS_COMMENT;
  /* GCC 9 correctly warns that this call to strpbrk is useless.  That
     is deliberate; this test is verifying that a side effect in an
     argument still occurs when the call itself is useless and could
     be optimized to return a constant.  */
  DIAG_IGNORE_NEEDS_COMMENT (9, "-Wunused-value");
  strpbrk (b++, "");
  DIAG_POP_NEEDS_COMMENT;
  if (b != a + 1)
    return 1;

  return 0;
}
