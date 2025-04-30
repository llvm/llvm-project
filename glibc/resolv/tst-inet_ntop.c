#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>

static int
do_test (void)
{
  struct in_addr addr4;
  struct in6_addr addr6;
  char buf[64];
  int result = 0;

  addr4.s_addr = 0xe0e0e0e0;
  addr6.s6_addr16[0] = 0;
  addr6.s6_addr16[1] = 0;
  addr6.s6_addr16[2] = 0;
  addr6.s6_addr16[3] = 0;
  addr6.s6_addr16[4] = 0;
  addr6.s6_addr16[5] = 0xffff;
  addr6.s6_addr32[3] = 0xe0e0e0e0;
  memset (buf, 'x', sizeof buf);

  if (inet_ntop (AF_INET, &addr4, buf, 15) != NULL)
    {
      puts ("1st inet_ntop returned non-NULL");
      result++;
    }
  else if (errno != ENOSPC)
    {
      puts ("1st inet_ntop didn't fail with ENOSPC");
      result++;
    }
  if (buf[15] != 'x')
    {
      puts ("1st inet_ntop wrote past the end of buffer");
      result++;
    }

  if (inet_ntop (AF_INET, &addr4, buf, 16) != buf)
    {
      puts ("2nd inet_ntop did not return buf");
      result++;
    }
  if (memcmp (buf, "224.224.224.224\0" "xxxxxxxx", 24) != 0)
    {
      puts ("2nd inet_ntop wrote past the end of buffer");
      result++;
    }

  if (inet_ntop (AF_INET6, &addr6, buf, 22) != NULL)
    {
      puts ("3rd inet_ntop returned non-NULL");
      result++;
    }
  else if (errno != ENOSPC)
    {
      puts ("3rd inet_ntop didn't fail with ENOSPC");
      result++;
    }
  if (buf[22] != 'x')
    {
      puts ("3rd inet_ntop wrote past the end of buffer");
      result++;
    }

  if (inet_ntop (AF_INET6, &addr6, buf, 23) != buf)
    {
      puts ("4th inet_ntop did not return buf");
      result++;
    }
  if (memcmp (buf, "::ffff:224.224.224.224\0" "xxxxxxxx", 31) != 0)
    {
      puts ("4th inet_ntop wrote past the end of buffer");
      result++;
    }

  memset (&addr6.s6_addr, 0xe0, sizeof (addr6.s6_addr));

  if (inet_ntop (AF_INET6, &addr6, buf, 39) != NULL)
    {
      puts ("5th inet_ntop returned non-NULL");
      result++;
    }
  else if (errno != ENOSPC)
    {
      puts ("5th inet_ntop didn't fail with ENOSPC");
      result++;
    }
  if (buf[39] != 'x')
    {
      puts ("5th inet_ntop wrote past the end of buffer");
      result++;
    }

  if (inet_ntop (AF_INET6, &addr6, buf, 40) != buf)
    {
      puts ("6th inet_ntop did not return buf");
      result++;
    }
  if (memcmp (buf, "e0e0:e0e0:e0e0:e0e0:e0e0:e0e0:e0e0:e0e0\0"
		   "xxxxxxxx", 48) != 0)
    {
      puts ("6th inet_ntop wrote past the end of buffer");
      result++;
    }


  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
