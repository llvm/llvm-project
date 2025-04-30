#include <stdio.h>



int
main (void)
{
  int res = 0;
  char buf[100];

#define TEST(nr, result, format, args...) \
  if (sprintf (buf, format, ## args) != result)				      \
    {									      \
      printf ("test %d failed (\"%s\",  %d)\n", nr, buf, result);	      \
      res = 1;								      \
    }

  TEST (1, 2, "%d", -1);
  TEST (2, 2, "% 2d", 1);
  TEST (3, 3, "%#x", 1);
  TEST (4, 2, "%+d", 1);
  TEST (5, 2, "% d", 1);
  TEST (6, 2, "%-d", -1);
  TEST (7, 2, "%- 2d", 1);
  TEST (8, 3, "%-#x", 1);
  TEST (9, 2, "%-+d", 1);
  TEST (10, 2, "%- d", 1);

  return res;
}
