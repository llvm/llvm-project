/* Test of perror.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.
   To be used only for testing glibc.  */

#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>


#define MB_EXP \
  "null mode test 1: Invalid or incomplete multibyte or wide character\n" \
  "multibyte string\n" \
  "<0 mode test: Invalid argument\n"
#define MB_EXP_LEN (sizeof (MB_EXP) - 1)

#define WC_EXP \
  "null mode test 2: Invalid or incomplete multibyte or wide character\n" \
  "wide string\n" \
  ">0 mode test: Invalid argument\n"
#define WC_EXP_LEN (sizeof (WC_EXP) - 1)


static int
do_test (void)
{
  int fd;
  char fname[] = "/tmp/tst-perror.XXXXXX";
  int result = 0;
  char buf[200];
  ssize_t n;

  fd = mkstemp (fname);
  if (fd == -1)
    error (EXIT_FAILURE, errno, "cannot create temporary file");

  /* Make sure the file gets removed.  */
  unlink (fname);

  fclose (stderr);

  if (dup2 (fd, 2) == -1)
    {
      printf ("cannot create file descriptor 2: %m\n");
      exit (EXIT_FAILURE);
    }

  stderr = fdopen (2, "w");
  if (stderr == NULL)
    {
      printf ("fdopen failed: %m\n");
      exit (EXIT_FAILURE);
    }

  if (fwide (stderr, 0) != 0)
    {
      printf ("stderr not initially in mode 0\n");
      exit (EXIT_FAILURE);
    }

  errno = EILSEQ;
  perror ("null mode test 1");

  if (fwide (stderr, 0) != 0)
    {
      puts ("perror changed the mode from 0");
      result = 1;
    }

  fputs ("multibyte string\n", stderr);

  if (fwide (stderr, 0) >= 0)
    {
      puts ("fputs didn't set orientation to narrow");
      result = 1;
    }

  errno = EINVAL;
  perror ("<0 mode test");

  fclose (stderr);

  lseek (fd, 0, SEEK_SET);
  n = read (fd, buf, sizeof (buf));
  if (n != MB_EXP_LEN || memcmp (buf, MB_EXP, MB_EXP_LEN) != 0)
    {
      printf ("multibyte test failed.  Expected:\n%s\nGot:\n%.*s\n",
	      MB_EXP, (int) n, buf);
      result = 1;
    }
  else
    puts ("multibyte test succeeded");

  lseek (fd, 0, SEEK_SET);
  ftruncate (fd, 0);

  if (dup2 (fd, 2) == -1)
    {
      printf ("cannot create file descriptor 2: %m\n");
      exit (EXIT_FAILURE);
    }
  stderr = fdopen (2, "w");
  if (stderr == NULL)
    {
      printf ("fdopen failed: %m\n");
      exit (EXIT_FAILURE);
    }

  if (fwide (stderr, 0) != 0)
    {
      printf ("stderr not initially in mode 0\n");
      exit (EXIT_FAILURE);
    }

  errno = EILSEQ;
  perror ("null mode test 2");

  if (fwide (stderr, 0) != 0)
    {
      puts ("perror changed the mode from 0");
      result = 1;
    }

  fputws (L"wide string\n", stderr);

  if (fwide (stderr, 0) <= 0)
    {
      puts ("fputws didn't set orientation to wide");
      result = 1;
    }

  errno = EINVAL;
  perror (">0 mode test");

  fclose (stderr);

  lseek (fd, 0, SEEK_SET);
  n = read (fd, buf, sizeof (buf));
  if (n != WC_EXP_LEN || memcmp (buf, WC_EXP, WC_EXP_LEN) != 0)
    {
      printf ("wide test failed.  Expected:\n%s\nGot:\n%.*s\n",
	      WC_EXP, (int) n, buf);
      result = 1;
    }
  else
    puts ("wide test succeeded");

  close (fd);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
