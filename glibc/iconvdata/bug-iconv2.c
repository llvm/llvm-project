/* Test case by Akira Higuchi <a@kondara.org>.  */

#include <stdio.h>
#include <stdlib.h>
#include <iconv.h>

int
main (void)
{
  const char *dummy_codesets[] =
  {
    "ISO_8859-1", "ISO_8859-2", "ISO_8859-3", "ISO_8859-4",
    "ISO_8859-5", "ISO_8859-6", "ISO_8859-7", "ISO_8859-8"
  };
  iconv_t dummy_cd[8], cd_a;
  int i;
  char buffer[1024], *to = buffer;
  char *from = (char *) "foobar";
  size_t to_left = 1024, from_left = 6;

  /* load dummy modules */
  for (i = 0; i < 8; i++)
    if ((dummy_cd[i] = iconv_open (dummy_codesets[i], "UTF8")) == (iconv_t) -1)
      exit (1);

  /* load a module... */
  if ((cd_a = iconv_open ("EUC-JP", "UTF8")) == (iconv_t) -1)
    exit (1);
  /* and close it once. we'll reload this later */
  iconv_close (cd_a);

  /* unload dummy modules */
  for (i = 0; i < 8; i++)
    iconv_close (dummy_cd[i]);

  /* load the module again */
  if ((cd_a = iconv_open ("EUC-JP", "UTF8")) == (iconv_t) -1)
    exit (1);

  puts ("This used to crash");
  printf ("%zd\n", iconv (cd_a, &from, &from_left, &to, &to_left));
  iconv_close (cd_a);

  puts ("works now");

  return 0;
}
