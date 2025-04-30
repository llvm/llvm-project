/* Copyright (c) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1998.

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

#include <errno.h>
#include <error.h>
#include <inttypes.h>
#include <langinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <libintl.h>

#include "nscd.h"
#include "dbg_log.h"
#include "selinux.h"
#ifdef HAVE_SELINUX
# include <selinux/selinux.h>
# include <selinux/avc.h>
#endif /* HAVE_SELINUX */

/* We use this to make sure the receiver is the same.  The lower 16
   bits are reserved for flags indicating compilation variants.  This
   version needs to be updated if the definition of struct statdata
   changes.  */
#define STATDATA_VERSION  0x01020000U

#ifdef HAVE_SELINUX
# define STATDATA_VERSION_SELINUX_FLAG 0x0001U
#else
# define STATDATA_VERSION_SELINUX_FLAG 0x0000U
#endif

/* All flags affecting the struct statdata layout.  */
#define STATDATA_VERSION_FLAGS STATDATA_VERSION_SELINUX_FLAG

/* The full version number for struct statdata.  */
#define STATDATA_VERSION_FULL (STATDATA_VERSION | STATDATA_VERSION_FLAGS)

/* Statistic data for one database.  */
struct dbstat
{
  int enabled;
  int check_file;
  int shared;
  int persistent;
  size_t module;

  unsigned long int postimeout;
  unsigned long int negtimeout;

  size_t nentries;
  size_t maxnentries;
  size_t maxnsearched;
  size_t datasize;
  size_t dataused;

  uintmax_t poshit;
  uintmax_t neghit;
  uintmax_t posmiss;
  uintmax_t negmiss;

  uintmax_t rdlockdelayed;
  uintmax_t wrlockdelayed;

  uintmax_t addfailed;
};

/* Record for transmitting statistics.  If this definition changes,
   update STATDATA_VERSION above.  */
struct statdata
{
  unsigned int version;		/* Must be STATDATA_VERSION_FULL.  */
  int debug_level;
  time_t runtime;
  unsigned long int client_queued;
  int nthreads;
  int max_nthreads;
  int paranoia;
  time_t restart_interval;
  unsigned int reload_count;
  int ndbs;
  struct dbstat dbs[lastdb];
#ifdef HAVE_SELINUX
  struct avc_cache_stats cstats;
#endif /* HAVE_SELINUX */
};


void
send_stats (int fd, struct database_dyn dbs[lastdb])
{
  struct statdata data;
  int cnt;

  memset (&data, 0, sizeof (data));

  data.version = STATDATA_VERSION_FULL;
  data.debug_level = debug_level;
  data.runtime = time (NULL) - start_time;
  data.client_queued = client_queued;
  data.nthreads = nthreads;
  data.max_nthreads = max_nthreads;
  data.paranoia = paranoia;
  data.restart_interval = restart_interval;
  data.reload_count = reload_count;
  data.ndbs = lastdb;

  for (cnt = 0; cnt < lastdb; ++cnt)
    {
      memset (&data.dbs[cnt], 0, sizeof (data.dbs[cnt]));
      data.dbs[cnt].enabled = dbs[cnt].enabled;
      data.dbs[cnt].check_file = dbs[cnt].check_file;
      data.dbs[cnt].shared = dbs[cnt].shared;
      data.dbs[cnt].persistent = dbs[cnt].persistent;
      data.dbs[cnt].postimeout = dbs[cnt].postimeout;
      data.dbs[cnt].negtimeout = dbs[cnt].negtimeout;
      if (dbs[cnt].head != NULL)
	{
	  data.dbs[cnt].module = dbs[cnt].head->module;
	  data.dbs[cnt].poshit = dbs[cnt].head->poshit;
	  data.dbs[cnt].neghit = dbs[cnt].head->neghit;
	  data.dbs[cnt].posmiss = dbs[cnt].head->posmiss;
	  data.dbs[cnt].negmiss = dbs[cnt].head->negmiss;
	  data.dbs[cnt].nentries = dbs[cnt].head->nentries;
	  data.dbs[cnt].maxnentries = dbs[cnt].head->maxnentries;
	  data.dbs[cnt].datasize = dbs[cnt].head->data_size;
	  data.dbs[cnt].dataused = dbs[cnt].head->first_free;
	  data.dbs[cnt].maxnsearched = dbs[cnt].head->maxnsearched;
	  data.dbs[cnt].rdlockdelayed = dbs[cnt].head->rdlockdelayed;
	  data.dbs[cnt].wrlockdelayed = dbs[cnt].head->wrlockdelayed;
	  data.dbs[cnt].addfailed = dbs[cnt].head->addfailed;
	}
    }

  if (selinux_enabled)
    nscd_avc_cache_stats (&data.cstats);

  if (TEMP_FAILURE_RETRY (send (fd, &data, sizeof (data), MSG_NOSIGNAL))
      != sizeof (data))
    {
      char buf[256];
      dbg_log (_("cannot write statistics: %s"),
	       strerror_r (errno, buf, sizeof (buf)));
    }
}


int
receive_print_stats (void)
{
  struct statdata data;
  request_header req;
  ssize_t nbytes;
  int fd;
  int i;
  uid_t uid = getuid ();
  const char *yesstr = _("yes");
  const char *nostr = _("no");

  /* Find out whether there is another user but root allowed to
     request statistics.  */
  if (uid != 0)
    {
      /* User specified?  */
      if(stat_user == NULL || stat_uid != uid)
	{
	  if (stat_user != NULL)
	    error (EXIT_FAILURE, 0,
		   _("Only root or %s is allowed to use this option!"),
		   stat_user);
	  else
	    error (EXIT_FAILURE, 0,
		   _("Only root is allowed to use this option!"));
	}
    }

  /* Open a socket to the running nscd.  */
  fd = nscd_open_socket ();
  if (fd == -1)
    error (EXIT_FAILURE, 0, _("nscd not running!\n"));

  /* Send the request.  */
  req.version = NSCD_VERSION;
  req.type = GETSTAT;
  req.key_len = 0;
  nbytes = TEMP_FAILURE_RETRY (send (fd, &req, sizeof (request_header),
				     MSG_NOSIGNAL));
  if (nbytes != sizeof (request_header))
    {
      int err = errno;
      close (fd);
      error (EXIT_FAILURE, err, _("write incomplete"));
    }

  /* Read as much data as we expect.  */
  if (TEMP_FAILURE_RETRY (read (fd, &data, sizeof (data))) != sizeof (data)
      || (data.version != STATDATA_VERSION_FULL
	  /* Yes, this is an assignment!  */
	  && (errno = EINVAL)))
    {
      /* Not the right version.  */
      int err = errno;
      close (fd);
      error (EXIT_FAILURE, err, _("cannot read statistics data"));
    }

  printf (_("nscd configuration:\n\n%15d  server debug level\n"),
	  data.debug_level);

  /* We know that we can simply subtract time_t values.  */
  unsigned long int diff = data.runtime;
  unsigned int ndays = 0;
  unsigned int nhours = 0;
  unsigned int nmins = 0;
  if (diff > 24 * 60 * 60)
    {
      ndays = diff / (24 * 60 * 60);
      diff %= 24 * 60 * 60;
    }
  if (diff > 60 * 60)
    {
      nhours = diff / (60 * 60);
      diff %= 60 * 60;
    }
  if (diff > 60)
    {
      nmins = diff / 60;
      diff %= 60;
    }
  if (ndays != 0)
    printf (_("%3ud %2uh %2um %2lus  server runtime\n"),
	    ndays, nhours, nmins, diff);
  else if (nhours != 0)
    printf (_("    %2uh %2um %2lus  server runtime\n"), nhours, nmins, diff);
  else if (nmins != 0)
    printf (_("        %2um %2lus  server runtime\n"), nmins, diff);
  else
    printf (_("            %2lus  server runtime\n"), diff);

  printf (_("%15d  current number of threads\n"
	    "%15d  maximum number of threads\n"
	    "%15lu  number of times clients had to wait\n"
	    "%15s  paranoia mode enabled\n"
	    "%15lu  restart internal\n"
	    "%15u  reload count\n"),
	  data.nthreads, data.max_nthreads, data.client_queued,
	  data.paranoia ? yesstr : nostr,
	  (unsigned long int) data.restart_interval, data.reload_count);

  for (i = 0; i < lastdb; ++i)
    {
      unsigned long int hit = data.dbs[i].poshit + data.dbs[i].neghit;
      unsigned long int all = hit + data.dbs[i].posmiss + data.dbs[i].negmiss;
      const char *enabled = data.dbs[i].enabled ? yesstr : nostr;
      const char *check_file = data.dbs[i].check_file ? yesstr : nostr;
      const char *shared = data.dbs[i].shared ? yesstr : nostr;
      const char *persistent = data.dbs[i].persistent ? yesstr : nostr;

      if (enabled[0] == '\0')
	/* The locale does not provide this information so we have to
	   translate it ourself.  Since we should avoid short translation
	   terms we artifically increase the length.  */
	enabled = data.dbs[i].enabled ? yesstr : nostr;
      if (check_file[0] == '\0')
	check_file = data.dbs[i].check_file ? yesstr : nostr;
      if (shared[0] == '\0')
	shared = data.dbs[i].shared ? yesstr : nostr;
      if (persistent[0] == '\0')
	persistent = data.dbs[i].persistent ? yesstr : nostr;

      if (all == 0)
	/* If nothing happened so far report a 0% hit rate.  */
	all = 1;

      printf (_("\n%s cache:\n\n"
		"%15s  cache is enabled\n"
		"%15s  cache is persistent\n"
		"%15s  cache is shared\n"
		"%15zu  suggested size\n"
		"%15zu  total data pool size\n"
		"%15zu  used data pool size\n"
		"%15lu  seconds time to live for positive entries\n"
		"%15lu  seconds time to live for negative entries\n"
		"%15" PRIuMAX "  cache hits on positive entries\n"
		"%15" PRIuMAX "  cache hits on negative entries\n"
		"%15" PRIuMAX "  cache misses on positive entries\n"
		"%15" PRIuMAX "  cache misses on negative entries\n"
		"%15lu%% cache hit rate\n"
		"%15zu  current number of cached values\n"
		"%15zu  maximum number of cached values\n"
		"%15zu  maximum chain length searched\n"
		"%15" PRIuMAX "  number of delays on rdlock\n"
		"%15" PRIuMAX "  number of delays on wrlock\n"
		"%15" PRIuMAX "  memory allocations failed\n"
		"%15s  check /etc/%s for changes\n"),
	      dbnames[i], enabled, persistent, shared,
	      data.dbs[i].module,
	      data.dbs[i].datasize, data.dbs[i].dataused,
	      data.dbs[i].postimeout, data.dbs[i].negtimeout,
	      data.dbs[i].poshit, data.dbs[i].neghit,
	      data.dbs[i].posmiss, data.dbs[i].negmiss,
	      (100 * hit) / all,
	      data.dbs[i].nentries, data.dbs[i].maxnentries,
	      data.dbs[i].maxnsearched,
	      data.dbs[i].rdlockdelayed,
	      data.dbs[i].wrlockdelayed,
	      data.dbs[i].addfailed, check_file, dbnames[i]);
    }

  if (selinux_enabled)
    nscd_avc_print_stats (&data.cstats);

  close (fd);

  exit (0);
}
