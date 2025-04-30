/* Test case by Yoshito Kawada <KAWADA@jp.ibm.com>.  */
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>

int
main (int argc, char *argv[])
{
  int a = 3;
  int fd;
  char name[] = "/tmp/wprintf.out.XXXXXX";
  FILE *fp;
  char buf[100];
  size_t len;
  int res = 0;

  fd = mkstemp (name);
  if (fd == -1)
    error (EXIT_FAILURE, errno, "cannot open temporary file");

  unlink (name);

  setlocale (LC_ALL, "en_US.UTF-8");

  fp = fdopen (dup (fd), "w");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "fdopen(,\"w\")");

  fwprintf (fp, L"test start");
  fwprintf (fp, L" int %d\n", a);

  /* String with precision.  */
  fwprintf (fp, L"1[%6.3s]\n", argv[1]);

  fclose (fp);

  fp = fdopen (dup (fd), "a");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "fdopen(,\"a\")");

  setvbuf (fp, NULL, _IONBF, 0);

  /* fwprintf to unbuffered stream.   */
  fwprintf (fp, L"hello.\n");

  fclose (fp);


  /* Now read it back in.  This time using multibyte functions.  */
  lseek (fd, SEEK_SET, 0);
  fp = fdopen (fd, "r");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "fdopen(,\"r\")");

  if (fgets (buf, sizeof buf, fp) != buf)
    error (EXIT_FAILURE, errno, "first fgets");
  len = strlen (buf);
  if (buf[len - 1] == '\n')
    --len;
  else
    {
      puts ("newline missing after first line");
      res = 1;
    }
  printf ("1st line: \"%.*s\" -> %s\n", (int) len, buf,
	  strncmp (buf, "test start int 3", len) == 0 ? "OK" : "FAIL");
  res |= strncmp (buf, "test start int 3", len) != 0;

  if (fgets (buf, sizeof buf, fp) != buf)
    error (EXIT_FAILURE, errno, "second fgets");
  len = strlen (buf);
  if (buf[len - 1] == '\n')
    --len;
  else
    {
      puts ("newline missing after second line");
      res = 1;
    }
  printf ("2nd line: \"%.*s\" -> %s\n", (int) len, buf,
	  strncmp (buf, "1[   Som]", len) == 0 ? "OK" : "FAIL");
  res |= strncmp (buf, "1[   Som]", len) != 0;

  if (fgets (buf, sizeof buf, fp) != buf)
    error (EXIT_FAILURE, errno, "third fgets");
  len = strlen (buf);
  if (buf[len - 1] == '\n')
    --len;
  else
    {
      puts ("newline missing after third line");
      res = 1;
    }
  printf ("3rd line: \"%.*s\" -> %s\n", (int) len, buf,
	  strncmp (buf, "hello.", len) == 0 ? "OK" : "FAIL");
  res |= strncmp (buf, "hello.", len) != 0;

  return res;
}
