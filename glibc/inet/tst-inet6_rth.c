#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/ip6.h>

static int
do_test (void)
{
  int res = 0;
  char buf[1000];
  void *p = inet6_rth_init (buf, 24, IPV6_RTHDR_TYPE_0, 0);
  if (p == NULL)
    {
      puts ("first inet6_rth_init failed");
      res = 1;
    }
  else if (inet6_rth_add (p, &in6addr_any) == 0)
    {
      puts ("first inet6_rth_add succeeded");
      res = 1;
    }

  p = inet6_rth_init (buf, 24, IPV6_RTHDR_TYPE_0, 1);
  if (p == NULL)
    {
      puts ("second inet6_rth_init failed");
      res = 1;
    }
  else if (inet6_rth_add (p, &in6addr_any) != 0)
    {
      puts ("second inet6_rth_add failed");
      res = 1;
    }

  for (int nseg = 4; nseg < 6; ++nseg)
    {
      printf ("nseg = %d\n", nseg);

      p = inet6_rth_init (buf, sizeof (buf), IPV6_RTHDR_TYPE_0, nseg);
      if (p == NULL)
	{
	  puts ("third inet6_rth_init failed");
	  res = 1;
	}
      else
	{
	  struct in6_addr tmp;
	  memset (&tmp, '\0', sizeof (tmp));

	  for (int i = 0; i < nseg; ++i)
	    {
	      tmp.s6_addr[0] = i;
	      if (inet6_rth_add (p, &tmp) != 0)
		{
		  printf ("call %d of third inet6_rth_add failed\n", i + 1);
		  res = 1;
		  goto out;
		}
	    }
	  ((struct ip6_rthdr0 *) p)->ip6r0_segleft = 0;
	  if (inet6_rth_segments (p) != nseg)
	    {
	      puts ("\
inet6_rth_segments returned wrong value after loop with third inet6_rth_add");
	      res = 1;
	      goto out;
	    }

          union
          {
            char buffer[1000];
            struct ip6_rthdr0 rthdr0;
          } buf2;
	  if (inet6_rth_reverse (p, buf2.buffer) != 0)
	    {
	      puts ("first inet6_rth_reverse call failed");
	      res = 1;
	      goto out;
	    }
	  if (buf2.rthdr0.ip6r0_segleft != nseg)
	    {
	      puts ("segleft after first inet6_rth_reverse wrong");
	      res = 1;
	    }

	  if (inet6_rth_segments (p) != inet6_rth_segments (buf2.buffer))
	    {
	      puts ("number of seconds after first inet6_rth_reverse differs");
	      res = 1;
	      goto out;
	    }

	  for (int i = 0; i < nseg; ++i)
	    {
	      struct in6_addr *addr = inet6_rth_getaddr (buf2.buffer, i);
	      if (addr == NULL)
		{
		  printf ("call %d of first inet6_rth_getaddr failed\n",
			  i + 1);
		  res = 1;
		}
	      else if (addr->s6_addr[0] != nseg - 1 - i
		       || memcmp (&addr->s6_addr[1], &in6addr_any.s6_addr[1],
				  sizeof (in6addr_any)
				  - sizeof (in6addr_any.s6_addr[0])) != 0)
		{
		  char addrbuf[100];
		  inet_ntop (AF_INET6, addr, addrbuf, sizeof (addrbuf));
		  printf ("\
address %d after first inet6_rth_reverse wrong (%s)\n",
			  i + 1, addrbuf);
		  res = 1;
		}
	    }
	out:
	  ;
	}

      p = inet6_rth_init (buf, sizeof (buf), IPV6_RTHDR_TYPE_0, nseg);
      if (p == NULL)
	{
	  puts ("fourth inet6_rth_init failed");
	  res = 1;
	}
      else
	{
	  struct in6_addr tmp;
	  memset (&tmp, '\0', sizeof (tmp));

	  for (int i = 0; i < nseg; ++i)
	    {
	      tmp.s6_addr[0] = i;
	      if (inet6_rth_add (p, &tmp) != 0)
		{
		  printf ("call %d of fourth inet6_rth_add failed\n", i + 1);
		  res = 1;
		  goto out2;
		}
	    }
	  ((struct ip6_rthdr0 *) p)->ip6r0_segleft = 0;
	  if (inet6_rth_segments (p) != nseg)
	    {
	      puts ("\
inet6_rth_segments returned wrong value after loop with fourth inet6_rth_add");
	      res = 1;
	      goto out2;
	    }

	  if (inet6_rth_reverse (p, p) != 0)
	    {
	      puts ("second inet6_rth_reverse call failed");
	      res = 1;
	      goto out2;
	    }
	  if (((struct ip6_rthdr0 *) p)->ip6r0_segleft != nseg)
	    {
	      puts ("segleft after second inet6_rth_reverse wrong");
	      res = 1;
	    }

	  for (int i = 0; i < nseg; ++i)
	    {
	      struct in6_addr *addr = inet6_rth_getaddr (p, i);
	      if (addr == NULL)
		{
		  printf ("call %d of second inet6_rth_getaddr failed\n",
			  i + 1);
		  res = 1;
		}
	      else if (addr->s6_addr[0] != nseg - 1 - i
		       || memcmp (&addr->s6_addr[1], &in6addr_any.s6_addr[1],
				  sizeof (in6addr_any)
				  - sizeof (in6addr_any.s6_addr[0])) != 0)
		{
		  char addrbuf[100];
		  inet_ntop (AF_INET6, addr, addrbuf, sizeof (addrbuf));
		  printf ("\
address %d after second inet6_rth_reverse wrong (%s)\n",
			  i + 1, addrbuf);
		  res = 1;
		}
	    }
	out2:
	  ;
	}
    }

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
