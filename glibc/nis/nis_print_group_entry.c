/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1997.

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

#include <alloca.h>
#include <string.h>
#include <libintl.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

void
nis_print_group_entry (const_nis_name group)
{
  if (group != NULL && group[0] != '\0')
    {
      size_t grouplen = strlen (group);
      char buf[grouplen + 50];
      char leafbuf[grouplen + 3];
      char domainbuf[grouplen + 3];
      nis_result *res;
      char *cp, *cp2;
      u_int i;

      cp = stpcpy (buf, nis_leaf_of_r (group, leafbuf, sizeof (leafbuf) - 1));
      cp = stpcpy (cp, ".groups_dir");
      cp2 = nis_domain_of_r (group, domainbuf, sizeof (domainbuf) - 1);
      if (cp2 != NULL && cp2[0] != '\0')
	{
	  *cp++ = '.';
	  stpcpy (cp, cp2);
	}
      res = nis_lookup (buf, FOLLOW_LINKS | EXPAND_NAME);

      if (res == NULL)
	return;

      if (NIS_RES_STATUS (res) != NIS_SUCCESS
	  || NIS_RES_NUMOBJ (res) != 1
	  || __type_of (NIS_RES_OBJECT (res)) != NIS_GROUP_OBJ)
	{
	  nis_freeresult (res);
	  return;
	}

      char *mem_exp[NIS_RES_NUMOBJ (res)];
      char *mem_imp[NIS_RES_NUMOBJ (res)];
      char *mem_rec[NIS_RES_NUMOBJ (res)];
      char *nomem_exp[NIS_RES_NUMOBJ (res)];
      char *nomem_imp[NIS_RES_NUMOBJ (res)];
      char *nomem_rec[NIS_RES_NUMOBJ (res)];
      unsigned long mem_exp_cnt = 0, mem_imp_cnt = 0, mem_rec_cnt = 0;
      unsigned long nomem_exp_cnt = 0, nomem_imp_cnt = 0, nomem_rec_cnt = 0;

      for (i = 0;
	   i < NIS_RES_OBJECT (res)->GR_data.gr_members.gr_members_len; ++i)
	{
	  char *grmem =
	    NIS_RES_OBJECT (res)->GR_data.gr_members.gr_members_val[i];
	  int neg = grmem[0] == '-';

	  switch (grmem[neg])
	    {
	    case '*':
	      if (neg)
		{
		  nomem_imp[nomem_imp_cnt] = grmem;
		  ++nomem_imp_cnt;
		}
	      else
		{
		  mem_imp[mem_imp_cnt] = grmem;
		  ++mem_imp_cnt;
		}
	      break;
	    case '@':
	      if (neg)
		{
		  nomem_rec[nomem_rec_cnt] = grmem;
		  ++nomem_rec_cnt;
		}
	      else
		{
		  mem_rec[mem_rec_cnt] = grmem;
		  ++mem_rec_cnt;
		}
	      break;
	    default:
	      if (neg)
		{
		  nomem_exp[nomem_exp_cnt] = grmem;
		  ++nomem_exp_cnt;
		}
	      else
		{
		  mem_exp[mem_exp_cnt] = grmem;
		  ++mem_exp_cnt;
		}
	      break;
	    }
	}
      {
	char buf[strlen (NIS_RES_OBJECT (res)->zo_domain) + 10];
	printf (_("Group entry for \"%s.%s\" group:\n"),
		NIS_RES_OBJECT (res)->zo_name,
		nis_domain_of_r (NIS_RES_OBJECT (res)->zo_domain,
				 buf, strlen (NIS_RES_OBJECT (res)->zo_domain)
				 + 10));
      }
      if (mem_exp_cnt)
	{
	  fputs (_("    Explicit members:\n"), stdout);
	  for (i = 0; i < mem_exp_cnt; ++i)
	    printf ("\t%s\n", mem_exp[i]);
	}
      else
	fputs (_("    No explicit members\n"), stdout);
      if (mem_imp_cnt)
	{
	  fputs (_("    Implicit members:\n"), stdout);
	  for (i = 0; i < mem_imp_cnt; ++i)
	    printf ("\t%s\n", &mem_imp[i][2]);
	}
      else
	fputs (_("    No implicit members\n"), stdout);
      if (mem_rec_cnt)
	{
	  fputs (_("    Recursive members:\n"), stdout);
	  for (i = 0; i < mem_rec_cnt; ++i)
	    printf ("\t%s\n", &mem_rec[i][1]);
	}
      else
        fputs (_("    No recursive members\n"), stdout);
      if (nomem_exp_cnt)
	{
	  fputs (_("    Explicit nonmembers:\n"), stdout);
	  for (i = 0; i < nomem_exp_cnt; ++i)
	    printf ("\t%s\n", &nomem_exp[i][1]);
	}
      else
	fputs (_("    No explicit nonmembers\n"), stdout);
      if (nomem_imp_cnt)
	{
	  fputs (_("    Implicit nonmembers:\n"), stdout);
	  for (i = 0; i < nomem_imp_cnt; ++i)
	    printf ("\t%s\n", &nomem_imp[i][3]);
	}
      else
	fputs (_("    No implicit nonmembers\n"), stdout);
      if (nomem_rec_cnt)
	{
	  fputs (_("    Recursive nonmembers:\n"), stdout);
	  for (i = 0; i < nomem_rec_cnt; ++i)
	    printf ("\t%s=n", &nomem_rec[i][2]);
	}
      else
        fputs (_("    No recursive nonmembers\n"), stdout);

      nis_freeresult (res);
    }
}
libnsl_hidden_nolink_def (nis_print_group_entry, GLIBC_2_1)
