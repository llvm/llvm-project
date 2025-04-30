#include <stdio.h>
#include <string.h>
#include <netinet/in.h>


static int
do_test (void)
{
  int result = 0;
  char buf[16];
  memset (buf, '\0', 16);

  if (! IN6_IS_ADDR_UNSPECIFIED (buf))
    {
      puts ("positive IN6_IS_ADDR_UNSPECIFIED failed");
      result = 1;
    }
  for (size_t i = 0; i < 16; ++i)
    {
      buf[i] = 1;
      if (IN6_IS_ADDR_UNSPECIFIED (buf))
	{
	  printf ("negative IN6_IS_ADDR_UNSPECIFIED with byte %zu failed\n",
		  i);
	  result = 1;
	}
      buf[i] = 0;
    }

  if (IN6_IS_ADDR_LOOPBACK (buf))
    {
      puts ("negative IN6_IS_ADDR_UNSPECIFIED failed");
      result = 1;
    }
  buf[15] = 1;
  if (! IN6_IS_ADDR_LOOPBACK (buf))
    {
      puts ("positive IN6_IS_ADDR_UNSPECIFIED failed");
      result = 1;
    }
  buf[15] = 0;

  buf[0] = 0xfe;
  buf[1] = 0x80;
  if (! IN6_IS_ADDR_LINKLOCAL (buf))
    {
      puts ("positive IN6_IS_ADDR_LINKLOCAL failed");
      result = 1;
    }
  for (size_t i = 1; i < 16; ++i)
    {
      buf[i] ^= 1;
      if (! IN6_IS_ADDR_LINKLOCAL (buf))
	{
	  printf ("positive IN6_IS_ADDR_LINKLOCAL byte %zu failed\n", i);
	  result = 1;
	}
      buf[i] ^= 1;
    }
  buf[0] = 0xff;
  buf[1] = 0x80;
  if (IN6_IS_ADDR_LINKLOCAL (buf))
    {
      puts ("negative IN6_IS_ADDR_LINKLOCAL failed");
      result = 1;
    }
  buf[0] = 0xfe;
  buf[1] = 0xc0;
  if (IN6_IS_ADDR_LINKLOCAL (buf))
    {
      puts ("negative IN6_IS_ADDR_LINKLOCAL #2 failed");
      result = 1;
    }

  buf[0] = 0xfe;
  buf[1] = 0xc0;
  if (! IN6_IS_ADDR_SITELOCAL (buf))
    {
      puts ("positive IN6_IS_ADDR_SITELOCAL failed");
      result = 1;
    }
  for (size_t i = 1; i < 16; ++i)
    {
      buf[i] ^= 1;
      if (! IN6_IS_ADDR_SITELOCAL (buf))
	{
	  printf ("positive IN6_IS_ADDR_SITELOCAL byte %zu failed\n", i);
	  result = 1;
	}
      buf[i] ^= 1;
    }
  buf[0] = 0xff;
  buf[1] = 0x80;
  if (IN6_IS_ADDR_SITELOCAL (buf))
    {
      puts ("negative IN6_IS_ADDR_SITELOCAL failed");
      result = 1;
    }
  buf[0] = 0xf8;
  buf[1] = 0xc0;
  if (IN6_IS_ADDR_SITELOCAL (buf))
    {
      puts ("negative IN6_IS_ADDR_SITELOCAL #2 failed");
      result = 1;
    }

  memset (buf, '\0', 16);
  buf[10] = 0xff;
  buf[11] = 0xff;
  if (! IN6_IS_ADDR_V4MAPPED (buf))
    {
      puts ("positive IN6_IS_ADDR_V4MAPPED failed");
      result = 1;
    }
  for (size_t i = 12; i < 16; ++i)
    {
      buf[i] ^= 1;
      if (! IN6_IS_ADDR_V4MAPPED (buf))
	{
	  printf ("positive IN6_IS_ADDR_V4MAPPED byte %zu failed\n", i);
	  result = 1;
	}
      buf[i] ^= 1;
    }
  for (size_t i = 0; i < 12; ++i)
    {
      buf[i] ^= 1;
      if (IN6_IS_ADDR_V4MAPPED (buf))
	{
	  printf ("negative IN6_IS_ADDR_V4MAPPED byte %zu failed\n", i);
	  result = 1;
	}
      buf[i] ^= 1;
    }

  memset (buf, '\0', 16);
  for (size_t i = 12; i < 16; ++i)
    {
      buf[i] ^= 2;
      if (! IN6_IS_ADDR_V4COMPAT (buf))
	{
	  printf ("positive IN6_IS_ADDR_V4COMPAT byte %zu failed\n", i);
	  result = 1;
	}
      buf[i] ^= 2;
    }
  for (size_t i = 0; i < 12; ++i)
    {
      buf[i] ^= 1;
      if (IN6_IS_ADDR_V4COMPAT (buf))
	{
	  printf ("negative IN6_IS_ADDR_V4COMPAT byte %zu failed\n", i);
	  result = 1;
	}
      buf[i] ^= 1;
    }
  if (IN6_IS_ADDR_V4COMPAT (buf))
    {
      puts ("negative IN6_IS_ADDR_V4COMPAT #2 failed");
      result = 1;
    }
  buf[15] = 1;
  if (IN6_IS_ADDR_V4COMPAT (buf))
    {
      puts ("negative IN6_IS_ADDR_V4COMPAT #3 failed");
      result = 1;
    }

  return result;
}

#include <support/test-driver.c>
