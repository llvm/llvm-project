/* Common code for NSS test cases.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */


/* There are two (or more) NSS test modules named nss_test1,
   nss_test2, etc.  Each one will call a function IN THE TEST CASE
   called _nss_test1_init_hook(test_tables *) (or _nss_test2_*, etc).

   In your copy of the hook function, you may change the *_table
   pointers in the passed struct to point to static tables in your
   test case, and the test modules will use that table instead.

   Your tables MUST end with an entry that has a *_LAST() macro.
   Use the *_ISLAST() macro to test for end of list.

   Use __nss_configure_lookup("passwd", "test1 test2") (for example) to
   configure NSS to use the test modules.  */

#include <pwd.h>
#include <grp.h>
#include <shadow.h>
#include <netdb.h>

typedef struct test_tables {
  struct passwd *pwd_table;
  struct group *grp_table;
  struct spwd *spwd_table;
  struct hostent *host_table;
} test_tables;

extern void _nss_test1_init_hook (test_tables *) __attribute__((weak));
extern void _nss_test2_init_hook (test_tables *) __attribute__((weak));

#define PWD_LAST()    { .pw_name = NULL, .pw_uid = 0 }
#define GRP_LAST()    { .gr_name = NULL, .gr_gid = 0 }
#define SPWD_LAST()    { .sp_namp = NULL, .sp_pwdp = NULL }
#define HOST_LAST()    { .h_name = NULL, .h_aliases = NULL, .h_length = 0, .h_addr_list = NULL }

#define PWD_ISLAST(p)    ((p)->pw_name == NULL && (p)->pw_uid == 0)
#define GRP_ISLAST(g)    ((g)->gr_name == NULL && (g)->gr_gid == 0)
#define SPWD_ISLAST(s)    ((s)->sp_namp == NULL && (s)->sp_pwdp == 0)
#define HOST_ISLAST(h)    ((h)->h_name == NULL && (h)->h_length == 0)

/* Macros to fill in the tables easily.  */

/* Note that the "unparameterized" fields are not magic; they're just
   arbitrary values.  Tests which need to verify those fields should
   fill them in explicitly.  */

#define PWD(u) \
    { .pw_name = (char *) "name" #u, .pw_passwd = (char *) "*", .pw_uid = u,  \
      .pw_gid = 100, .pw_gecos = (char *) "*", .pw_dir = (char *) "*",	      \
      .pw_shell = (char *) "*" }

#define PWD_N(u,n)								\
    { .pw_name = (char *) n, .pw_passwd = (char *) "*", .pw_uid = u,  \
      .pw_gid = 100, .pw_gecos = (char *) "*", .pw_dir = (char *) "*",	      \
      .pw_shell = (char *) "*" }

#define GRP(u) \
    { .gr_name = (char *) "name" #u, .gr_passwd = (char *) "*", .gr_gid = u, \
      .gr_mem = (char **) group_##u }

#define GRP_N(u,n,m)						     \
    { .gr_name = (char *) n, .gr_passwd = (char *) "*", .gr_gid = u, \
      .gr_mem = (char **) m }

#define SPWD(u) \
    { .sp_namp = (char *) "name" #u, .sp_pwdp = (char *) "passwd" #u }

#define HOST(u)								\
    { .h_name = (char *) "name" #u, .h_aliases = NULL, .h_addrtype = u,	\
      .h_length = 4,							\
      .h_addr_list = (char **) hostaddr_##u  }

/*------------------------------------------------------------*/

/* Helper functions for testing passwd entries.  Call
   compare_passwds() passing a test index, the passwd entry you got,
   and the expected passwd entry.  The function will return the number
   of mismatches, or zero of the two records are the same.  */

static void __attribute__((used))
print_passwd (struct passwd *p)
{
  printf ("    passwd %u.%s (%s) :", p->pw_uid, p->pw_name, p->pw_passwd);
  printf (" %u, %s, %s, %s\n", p->pw_gid, p->pw_gecos, p->pw_dir, p->pw_shell);
  printf ("\n");
}

static int  __attribute__((used))
compare_passwd_field (int i, struct passwd *p, const char *got,
		      const char *exp, const char *name)
{
  /* Does the entry have a value?  */
  if (got == NULL)
    {
      printf ("[%d] passwd %s for %u.%s was (null)\n",
	      i, name,
	      p->pw_uid, p->pw_name);
      return 1;
    }
  /* Does the entry have an unexpected name?  */
  else if (exp == NULL)
    {
      printf ("[%d] passwd %s for %u.(null) was %s\n",
	      i, name,
	      p->pw_uid, got);
      return 1;
    }
  /* And is it correct?  */
  else if (got && strcmp (got, exp) != 0)
    {
      printf("[%d] passwd entry %u.%s had %s \"%s\" (expected \"%s\") \n",
	     i,
	     p->pw_uid, p->pw_name, name,
	     got, exp);
      return 1;
    }
  return 0;
}

#define COMPARE_PWD_FIELD(f) \
  retval += compare_passwd_field (i, e, p->f, e->f, #f)

/* Compare passwd to expected passwd, return number of "problems".
   "I" is the index into the testcase data.  */
static int  __attribute__((used))
compare_passwds (int i, struct passwd *p, struct passwd *e)
{
  int retval = 0;

  /* Did we get the expected uid?  */
  if (p->pw_uid != e->pw_uid)
    {
      printf("[%d] passwd entry %u.%s had uid %u\n", i,
	     e->pw_uid, e->pw_name,
	     p->pw_uid);
      ++retval;
    }

  /* Did we get the expected gid?  */
  if (p->pw_gid != e->pw_gid)
    {
      printf("[%d] passwd entry %u.%s had gid %u (expected %u)\n", i,
	     e->pw_uid, e->pw_name,
	     p->pw_gid, e->pw_gid);
      ++retval;
    }

  COMPARE_PWD_FIELD (pw_name);
  COMPARE_PWD_FIELD (pw_passwd);
  COMPARE_PWD_FIELD (pw_gecos);
  COMPARE_PWD_FIELD (pw_dir);
  COMPARE_PWD_FIELD (pw_shell);

  if (retval > 0)
    {
      /* Left in for debugging later, if needed.  */
      print_passwd (p);
      print_passwd (e);
    }

  return retval;
}

/*------------------------------------------------------------*/

/* Helpers for checking group entries.  See passwd helper comment
   above for details.  */

static void __attribute__((used))
print_group (struct group *g)
{
  int j;

  printf ("    group %u.%s (%s) :", g->gr_gid, g->gr_name, g->gr_passwd);
  if (g->gr_mem)
    for (j=0; g->gr_mem[j]; j++)
      printf ("%s%s", j==0 ? " " : ", ", g->gr_mem[j]);
  printf ("\n");
}

/* Compare group to expected group, return number of "problems".  "I"
   is the index into the testcase data.  */
static int  __attribute__((used))
compare_groups (int i, struct group *g, struct group *e)
{
  int j;
  int retval = 0;

  /* Did we get the expected gid?  */
  if (g->gr_gid != e->gr_gid)
    {
      printf("[%d] group entry %u.%s had gid %u\n", i,
	     e->gr_gid, e->gr_name,
	     g->gr_gid);
      ++retval;
    }

  /* Does the entry have a name?  */
  if (g->gr_name == NULL)
    {
      printf ("[%d] group name for %u.%s was (null)\n", i,
	      e->gr_gid, e->gr_name);
      ++retval;
    }
  /* Does the entry have an unexpected name?  */
  else if (e->gr_name == NULL)
    {
      printf ("[%d] group name for %u.(null) was %s\n", i,
	      e->gr_gid, g->gr_name);
      ++retval;
    }
  /* And is it correct?  */
  else if (strcmp (g->gr_name, e->gr_name) != 0)
    {
      printf("[%d] group entry %u.%s had name \"%s\"\n", i,
	     e->gr_gid, e->gr_name,
	     g->gr_name);
      ++retval;
    }

  /* Does the entry have a password?  */
  if (g->gr_passwd == NULL && e->gr_passwd != NULL)
    {
      printf ("[%d] group password for %u.%s was NULL\n", i,
	      e->gr_gid, e->gr_name);
      ++retval;
    }
  else if (g->gr_passwd != NULL && e->gr_passwd == NULL)
    {
      printf ("[%d] group password for %u.%s was not NULL\n", i,
	      e->gr_gid, e->gr_name);
      ++retval;
    }
  /* And is it correct?  */
  else if (g->gr_passwd && strcmp (g->gr_passwd, e->gr_passwd) != 0)
    {
      printf("[%d] group entry %u.%s had password \"%s\" (not \"%s\")\n", i,
	     e->gr_gid, e->gr_name,
	     g->gr_passwd, e->gr_passwd);
      ++retval;
    }

  /* Now compare group members... */

  if (e->gr_mem != NULL && g->gr_mem == NULL)
    {
      printf("[%d] group entry %u.%s missing member list\n", i,
	     e->gr_gid, e->gr_name);
      ++retval;
    }
  else if (e->gr_mem == NULL && g->gr_mem != NULL)
    {
      printf("[%d] group entry %u.%s has unexpected member list\n", i,
	     e->gr_gid, e->gr_name);
      ++retval;
    }
  else if (e->gr_mem == NULL && g->gr_mem == NULL)
    {
      /* This case is OK.  */
    }
  else
    {
      /* Compare two existing lists.  */
      j = 0;
      for (;;)
	{
	  if (g->gr_mem[j] == NULL && e->gr_mem[j] == NULL)
	    {
	      /* Matching end-of-lists.  */
	      break;
	    }
	  if (g->gr_mem[j] == NULL)
	    {
	      printf ("[%d] group member list for %u.%s is too short.\n", i,
		      e->gr_gid, e->gr_name);
	      ++retval;
	      break;
	    }
	  if (e->gr_mem[j] == NULL)
	    {
	      printf ("[%d] group member list for %u.%s is too long.\n", i,
		      e->gr_gid, e->gr_name);
	      ++retval;
	      break;
	    }
	  if (strcmp (g->gr_mem[j], e->gr_mem[j]) != 0)
	    {
	      printf ("[%d] group member list for %u.%s differs: %s vs %s.\n", i,
		      e->gr_gid, e->gr_name,
		      e->gr_mem[j], g->gr_mem[j]);
	      ++retval;
	    }

	  j++;
	}
    }

  if (retval > 0)
    {
      /* Left in for debugging later, if needed.  */
      print_group (g);
      print_group (e);
    }

  return retval;
}
