#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>


int
main (void)
{
  FILE *fp;
  const char teststring[] = "hello world";
  char buf[3072];
  int result = 0;
  char readbuf[256];

  /* Open a file.  */
  fp = tmpfile ();

  /* Set a buffer.  */
  if (setvbuf (fp, buf, _IOFBF, sizeof buf) == EOF)
    {
      printf ("setvbuf failed: %m\n");
      exit (1);
    }

  /* Get the buffer size.  */
  if (__fbufsize (fp) != sizeof buf)
    {
      printf ("__fbusize() reported a buffer size of %Zd bytes;"
	      " we installed a buffer with %Zd bytes\n",
	      __fbufsize (fp), sizeof buf);
      result = 1;
    }

  /* Write something and read it back.  */
  if (fputs (teststring, fp) == EOF)
    {
      printf ("writing to new stream failed: %m\n");
      exit (1);
    }
  rewind (fp);
  if (fgets (readbuf, sizeof readbuf, fp) == NULL)
    {
      printf ("reading from new stream failed: %m\n");
      exit (1);
    }
  if (strcmp (readbuf, teststring) != 0)
    {
      puts ("not the correct string read");
      exit (1);
    }

  /* The file must be opened for reading and writing.  */
  if (__freading (fp) == 0)
    {
      puts ("__freading() reported stream is not last read from");
      result = 1;
    }
  if (__fwriting (fp) != 0)
    {
      puts ("__fwriting() reported stream is write-only or last written to");
      result = 1;
    }
  rewind (fp);
  if (fputs (teststring, fp) == EOF)
    {
      printf ("writing(2) to new stream failed: %m\n");
      exit (1);
    }
  if (__fwriting (fp) == 0)
    {
      puts ("__fwriting() doe snot reported stream is last written to");
      result = 1;
    }
  if (__freading (fp) != 0)
    {
      puts ("__freading() reported stream is last read from");
      result = 1;
    }

  if (__freadable (fp) == 0)
    {
      puts ("__freading() reported stream is last readable");
      result = 1;
    }
  if (__fwritable (fp) == 0)
    {
      puts ("__freading() reported stream is last writable");
      result = 1;
    }

  /* The string we wrote above should still be in the buffer.  */
  if (__fpending (fp) != strlen (teststring))
    {
      printf ("__fpending() returned %Zd; expected %Zd\n",
	      __fpending (fp), strlen (teststring));
      result = 1;
    }
  /* Discard all the output.  */
  __fpurge (fp);
  /* And check again.  */
  if (__fpending (fp) != 0)
    {
      printf ("__fpending() returned %Zd; expected 0\n",
	      __fpending (fp));
      result = 1;
    }


  /* Find out whether buffer is line buffered.  */
  if (__flbf (fp) != 0)
    {
      puts ("__flbf() reports line buffered but it is fully buffered");
      result = 1;
    }

  if (setvbuf (fp, buf, _IOLBF, sizeof buf) == EOF)
    {
      printf ("setvbuf(2) failed: %m\n");
      exit (1);
    }
  if (__flbf (fp) == 0)
    {
      puts ("__flbf() reports file is not line buffered");
      result = 1;
    }

  if (setvbuf (fp, NULL, _IONBF, 0) == EOF)
    {
      printf ("setvbuf(3) failed: %m\n");
      exit (1);
    }
  if (__flbf (fp) != 0)
    {
      puts ("__flbf() reports line buffered but it is not buffered");
      result = 1;
    }

  fclose (fp);

  return result;
}
