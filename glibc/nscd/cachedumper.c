/* Copyright (c) 2020-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

/* cachedumper - dump a human-readable representation of a cache file.  */

#include <ctype.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <libintl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <arpa/inet.h>
#include <getopt.h>
#include <sys/param.h>

#include "nscd.h"
#include "dbg_log.h"

static void *the_cache;

#define NO_REF ((ref_t) -1)

/* Given a chunk of raw data CP of length LEN, print it in a hopefully
   user-readable format, including colorizing non-readable characters.
   STR prefixes it, if non-NULL.  If LEN is -1, CP is
   NUL-terminated.  */
unsigned char *
data_string (unsigned char *cp, const char *str, int len)
{
  int oops = 0;
  unsigned char *cpe = cp + len;
  printf ("%s", str);
  while (len == -1 || cp < cpe)
    {
      if (isgraph (*cp))
	putchar (*cp);
      else
	printf ("\033[%dm<%02x>\033[0m", *cp % 6 + 31, *cp);
      if (len == -1 && *cp == 0)
	return cp + 1;

      ++cp;
      if (++oops > 1000)
	break;
    }
  return cp;
}

void
nscd_print_cache (const char *name)
{
  struct stat st;
  int fd;
  int i;

  if (stat (name, &st) < 0)
    {
      perror (name);
      exit (1);
    }

  fd = open (name, O_RDONLY);

  the_cache = mmap (NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

  struct database_pers_head *dps = (struct database_pers_head *) the_cache;

  /* Shortcut for "print the cache offset (address) of X in the
     cache".  */
#define A(x) (int) ((char *) &(x) - (char *) the_cache)

  /* Common code for "print field DPS->F, it's offset, and contents".  */
#define DPS(f) printf("%08x: %24s : %10d %08x\n", A (dps->f), #f, (int) dps->f, (int) dps->f);

  if (debug_level > 0)
    {
      DPS (version);
      DPS (header_size);
      DPS (gc_cycle);
      DPS (nscd_certainly_running);
      DPS (timestamp);
      DPS (module);
      DPS (data_size);
      DPS (first_free);
      DPS (nentries);
      DPS (maxnentries);
      DPS (maxnsearched);
      DPS (poshit);
      DPS (neghit);
      DPS (posmiss);
      DPS (negmiss);
      DPS (rdlockdelayed);
      DPS (wrlockdelayed);
      DPS (addfailed);
      printf ("\n");
    }


  char *data = (char *) &dps->array[roundup (dps->module,
					     ALIGN / sizeof (ref_t))];

  /* Loop through each entry in the hash table, which is of size
     dps->module.  Raw data is stored after the hash table in the
     cache file.  */
  for (i = 0; i < dps->module; i++)
    {
      ref_t r = dps->array[i];
      if (r == NO_REF)
	continue;

      if (debug_level > 2)
	printf ("hash[%4d] = 0x%x\n", i, r);

      while (r != NO_REF)
	{
	  struct hashentry *here = (struct hashentry *) (data + r);

	  unsigned char *key = (unsigned char *) data + here->key;

	  printf ("\n%08x: type %s key %p \"", A (*here),
		  serv2str[here->type], key);

	  data_string (key, "", here->len);

	  struct datahead *dh = (struct datahead *) (data + here->packet);
	  printf ("\" (len:%ld)  Data %08lx\n", (long) here->len,
		  (long unsigned int) ((char *) dh - (char *) the_cache));

	  if (debug_level > 0)
	    {
/* Common code for printing fields in struct DATAHEAD DH.  */
#define DH(f) printf ("%08x; %24s : %10d %08x\n", A (dh->f), #f, (int) dh->f, (int) dh->f);
	      DH (allocsize);
	      DH (recsize);
	      DH (timeout);
	      DH (notfound);
	      DH (nreloads);
	      DH (usable);
	      DH (unused);
	      DH (ttl);
	    }

	  unsigned char *cp = (unsigned char *) (&dh->data[0]);
	  unsigned char *cpe =
	    (unsigned char *) (&dh->data[0]) + dh->allocsize;


	  int i;
	  uint32_t *grplens;

	  if (debug_level > 1)
	    {
	      data_string (cp, _(" - all data: "), cpe - cp);
	      printf ("\n");
	    }

	  /* These two are common to all responses.  */
	  printf ("V%d F%d",
		  dh->data[0].pwdata.version, dh->data[0].pwdata.found);

/* Shortcut for the common case where we iterate through
   fixed-length strings stored in the data portion of the
   cache.  CP is updated to point to the next string.  */
#define DSTR(str, l) cp = data_string (cp, str, l)

	  switch (here->type)
	    {
	    case GETPWBYNAME:
	    case GETPWBYUID:
	      {
		pw_response_header *pw = &(dh->data[0].pwdata);
		cp += sizeof (*pw);
		DSTR (" name ", pw->pw_name_len);
		DSTR (" passwd ", pw->pw_passwd_len);
		printf (" uid %d gid %d", pw->pw_uid, pw->pw_gid);
		DSTR (" gecos ", pw->pw_gecos_len);
		DSTR (" dir ", pw->pw_dir_len);
		DSTR (" shell ", pw->pw_shell_len);
		DSTR (" byuid ", -1);
		DSTR (" key ", -1);
		printf ("\n");
	      }
	      break;

	    case GETGRBYNAME:
	    case GETGRBYGID:
	      {
		gr_response_header *gr = &(dh->data[0].grdata);
		cp += sizeof (*gr);
		grplens = (uint32_t *) cp;
		cp += gr->gr_mem_cnt * sizeof (uint32_t);
		DSTR (" name ", gr->gr_name_len);
		DSTR (" passwd ", gr->gr_passwd_len);
		printf (" gid %d members %d [ ", (int) gr->gr_gid,
			(int) gr->gr_mem_cnt);
		for (i = 0; i < gr->gr_mem_cnt; i++)
		  DSTR (" ", grplens[i]);
		DSTR (" ] bygid ", -1);
		DSTR (" key ", -1);
		printf ("\n");
	      }
	      break;

	    case GETHOSTBYADDR:
	    case GETHOSTBYADDRv6:
	    case GETHOSTBYNAME:
	    case GETHOSTBYNAMEv6:
	      {
		hst_response_header *hst = &(dh->data[0].hstdata);
		printf (" addrtype %d error %d", hst->h_addrtype, hst->error);
		cp += sizeof (*hst);
		DSTR (" name ", hst->h_name_len);
		uint32_t *aliases_len = (uint32_t *) cp;
		cp += hst->h_aliases_cnt * sizeof (uint32_t);
		uint32_t *addrs = (uint32_t *) cp;
		cp += hst->h_length * hst->h_addr_list_cnt;

		if (hst->h_aliases_cnt)
		  {
		    printf (" aliases [");
		    for (i = 0; i < hst->h_aliases_cnt; i++)
		      DSTR (" ", aliases_len[i]);
		    printf (" ]");
		  }
		if (hst->h_addr_list_cnt)
		  {
		    char buf[INET6_ADDRSTRLEN];
		    printf (" addresses [");
		    for (i = 0; i < hst->h_addr_list_cnt; i++)
		      {
			inet_ntop (hst->h_addrtype, addrs, buf, sizeof (buf));
			printf (" %s", buf);
			addrs += hst->h_length;
		      }
		    printf (" ]");
		  }

		printf ("\n");
	      }
	      break;

	    case GETAI:
	      {
		ai_response_header *ai = &(dh->data[0].aidata);
		printf (" naddrs %ld addrslen %ld canonlen %ld error %d [",
			(long) ai->naddrs, (long) ai->addrslen,
			(long) ai->canonlen, ai->error);
		cp += sizeof (*ai);
		unsigned char *addrs = cp;
		unsigned char *families = cp + ai->addrslen;
		cp = families + ai->naddrs;
		char buf[INET6_ADDRSTRLEN];

		for (i = 0; i < ai->naddrs; i++)
		  {
		    switch (*families)
		      {
		      case AF_INET:
			inet_ntop (*families, addrs, buf, sizeof (buf));
			printf (" %s", buf);
			addrs += 4;
			break;
		      case AF_INET6:
			inet_ntop (*families, addrs, buf, sizeof (buf));
			printf (" %s", buf);
			addrs += 16;
			break;
		      }
		    families++;
		  }
		DSTR (" ] canon ", ai->canonlen);
		DSTR (" key ", -1);
		printf ("\n");
	      }
	      break;

	    case INITGROUPS:
	      {
		initgr_response_header *ig = &(dh->data[0].initgrdata);
		printf (" nresults %d groups [", (int) ig->ngrps);
		cp += sizeof (*ig);
		grplens = (uint32_t *) cp;
		cp += ig->ngrps * sizeof (uint32_t);
		for (i = 0; i < ig->ngrps; i++)
		  printf (" %d", grplens[i]);
		DSTR (" ] key ", -1);
		printf ("\n");
	      }
	      break;

	    case GETSERVBYNAME:
	    case GETSERVBYPORT:
	      {
		serv_response_header *serv = &(dh->data[0].servdata);
		printf (" alias_cnt %ld port %d (stored as %d)",
			(long) serv->s_aliases_cnt,
			((serv->s_port & 0xff00) >> 8) | ((serv->
							   s_port & 0xff) <<
							  8), serv->s_port);
		cp += sizeof (*serv);
		DSTR (" name ", serv->s_name_len);
		DSTR (" proto ", serv->s_proto_len);
		if (serv->s_aliases_cnt)
		  {
		    uint32_t *alias_len = (uint32_t *) cp;
		    printf (" aliases [");
		    cp += sizeof (uint32_t) * serv->s_aliases_cnt;
		    for (i = 0; i < serv->s_aliases_cnt; i++)
		      DSTR (" ", alias_len[i]);
		    printf (" ]");
		  }
		printf ("\n");
	      }
	      break;

	    case GETNETGRENT:
	      {
		netgroup_response_header *ng = &(dh->data[0].netgroupdata);
		printf (" nresults %d len %d\n",
			(int) ng->nresults, (int) ng->result_len);
		cp += sizeof (*ng);
		for (i = 0; i < ng->nresults; i++)
		  {
		    DSTR (" (", -1);
		    DSTR (",", -1);
		    DSTR (",", -1);
		    printf (")");
		  }
		printf ("\n");
	      }
	      break;

	    case INNETGR:
	      {
		innetgroup_response_header *ing =
		  &(dh->data[0].innetgroupdata);
		printf (" result %d\n", ing->result);
	      }
	      break;

	    default:
	      break;
	    }

	  if (debug_level > 2 && cp && cp < cpe)
	    {
	      printf (_(" - remaining data %p: "), cp);
	      data_string (cp, "", cpe - cp);
	      printf ("\n");
	    }


	  r = here->next;
	}
    }

  munmap (the_cache, st.st_size);

  exit (0);
}
