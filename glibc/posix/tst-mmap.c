#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>


static int
do_test (void)
{
  int result = 0;
  FILE *fp;
  size_t c;
  char buf[1000];
  int fd;
  unsigned char *ptr;
  size_t ps = sysconf (_SC_PAGESIZE);
  void *mem;

  /* Create a file and put some data in it.  */
  fp = tmpfile ();
  if (fp == NULL)
    {
      printf ("Cannot create temporary file: %m\n");
      return 1;
    }
  fd = fileno (fp);

  for (c = 0; c < sizeof (buf); ++c)
    buf[c] = '0' + (c % 10);

  for (c = 0; c < (ps * 4) / sizeof (buf); ++c)
    if (fwrite (buf, 1, sizeof (buf), fp) != sizeof (buf))
      {
	printf ("`fwrite' failed: %m\n");
	return 1;
      }
  fflush (fp);
  assert (ps + 1000 < c * sizeof (buf));

  /* First try something which is not allowed: map at an offset which is
     not modulo the pagesize.  */
  ptr = mmap (NULL, 1000, PROT_READ, MAP_SHARED, fd, ps - 1);
  if (ptr != MAP_FAILED)
    {
      puts ("mapping at offset with mod pagesize != 0 succeeded!");
      result = 1;
    }
  else if (errno != EINVAL && errno != ENOSYS)
    {
      puts ("wrong error value for mapping at offset with mod pagesize != 0: %m (should be EINVAL)");
      result = 1;
    }

  /* Try the same for mmap64.  */
  ptr = mmap64 (NULL, 1000, PROT_READ, MAP_SHARED, fd, ps - 1);
  if (ptr != MAP_FAILED)
    {
      puts ("mapping at offset with mod pagesize != 0 succeeded!");
      result = 1;
    }
  else if (errno != EINVAL && errno != ENOSYS)
    {
      puts ("wrong error value for mapping at offset with mod pagesize != 0: %m (should be EINVAL)");
      result = 1;
    }

  /* And the same for private mapping.  */
  ptr = mmap (NULL, 1000, PROT_READ, MAP_PRIVATE, fd, ps - 1);
  if (ptr != MAP_FAILED)
    {
      puts ("mapping at offset with mod pagesize != 0 succeeded!");
      result = 1;
    }
  else if (errno != EINVAL && errno != ENOSYS)
    {
      puts ("wrong error value for mapping at offset with mod pagesize != 0: %m (should be EINVAL)");
      result = 1;
    }

  /* Try the same for mmap64.  */
  ptr = mmap64 (NULL, 1000, PROT_READ, MAP_PRIVATE, fd, ps - 1);
  if (ptr != MAP_FAILED)
    {
      puts ("mapping at offset with mod pagesize != 0 succeeded!");
      result = 1;
    }
  else if (errno != EINVAL && errno != ENOSYS)
    {
      puts ("wrong error value for mapping at offset with mod pagesize != 0: %m (should be EINVAL)");
      result = 1;
    }

  /* Get a valid address.  */
  mem = malloc (2 * ps);
  if (mem != NULL)
    {
      /* Now we map at an address which is not mod pagesize.  */
      ptr = mmap (mem + 1, 1000, PROT_READ, MAP_SHARED | MAP_FIXED, fd, ps);
      if (ptr != MAP_FAILED)
	{
	  puts ("mapping at address with mod pagesize != 0 succeeded!");
	  result = 1;
	}
      else  if (errno != EINVAL && errno != ENOSYS)
	{
	  puts ("wrong error value for mapping at address with mod pagesize != 0: %m (should be EINVAL)");
	  result = 1;
	}

      /* Try the same for mmap64.  */
      ptr = mmap64 (mem + 1, 1000, PROT_READ, MAP_SHARED | MAP_FIXED, fd, ps);
      if (ptr != MAP_FAILED)
	{
	  puts ("mapping at address with mod pagesize != 0 succeeded!");
	  result = 1;
	}
      else  if (errno != EINVAL && errno != ENOSYS)
	{
	  puts ("wrong error value for mapping at address with mod pagesize != 0: %m (should be EINVAL)");
	  result = 1;
	}

      /* And again for MAP_PRIVATE.  */
      ptr = mmap (mem + 1, 1000, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, ps);
      if (ptr != MAP_FAILED)
	{
	  puts ("mapping at address with mod pagesize != 0 succeeded!");
	  result = 1;
	}
      else  if (errno != EINVAL && errno != ENOSYS)
	{
	  puts ("wrong error value for mapping at address with mod pagesize != 0: %m (should be EINVAL)");
	  result = 1;
	}

      /* Try the same for mmap64.  */
      ptr = mmap64 (mem + 1, 1000, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, ps);
      if (ptr != MAP_FAILED)
	{
	  puts ("mapping at address with mod pagesize != 0 succeeded!");
	  result = 1;
	}
      else  if (errno != EINVAL && errno != ENOSYS)
	{
	  puts ("wrong error value for mapping at address with mod pagesize != 0: %m (should be EINVAL)");
	  result = 1;
	}

      free (mem);
    }

  /* Now map the memory and see whether the content of the mapped area
     is correct.  */
  ptr = mmap (NULL, 1000, PROT_READ, MAP_SHARED, fd, ps);
  if (ptr == MAP_FAILED)
    {
      if (errno != ENOSYS)
	{
	  printf ("cannot mmap file: %m\n");
	  result = 1;
	}
    }
  else
    {
      for (c = ps; c < ps + 1000; ++c)
	if (ptr[c - ps] != '0' + (c % 10))
	  {
	    printf ("wrong data mapped at offset %zd\n", c);
	    result = 1;
	  }
    }

  /* And for mmap64. */
  ptr = mmap64 (NULL, 1000, PROT_READ, MAP_SHARED, fd, ps);
  if (ptr == MAP_FAILED)
    {
      if (errno != ENOSYS)
	{
	  printf ("cannot mmap file: %m\n");
	  result = 1;
	}
    }
  else
    {
      for (c = ps; c < ps + 1000; ++c)
	if (ptr[c - ps] != '0' + (c % 10))
	  {
	    printf ("wrong data mapped at offset %zd\n", c);
	    result = 1;
	  }
    }

  /* That's it.  */
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
