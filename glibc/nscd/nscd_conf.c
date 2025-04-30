/* Copyright (c) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de>, 1998.

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

#include <ctype.h>
#include <errno.h>
#include <error.h>
#include <libintl.h>
#include <malloc.h>
#include <pwd.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/types.h>

#include "dbg_log.h"
#include "nscd.h"


/* Names of the databases.  */
const char *const dbnames[lastdb] =
{
  [pwddb] = "passwd",
  [grpdb] = "group",
  [hstdb] = "hosts",
  [servdb] = "services",
  [netgrdb] = "netgroup"
};


static int
find_db (const char *name)
{
  for (int cnt = 0; cnt < lastdb; ++cnt)
    if (strcmp (name, dbnames[cnt]) == 0)
      return cnt;

  error (0, 0, _("database %s is not supported"), name);
  return -1;
}

int
nscd_parse_file (const char *fname, struct database_dyn dbs[lastdb])
{
  FILE *fp;
  char *line, *cp, *entry, *arg1, *arg2;
  size_t len;
  int cnt;
  const unsigned int initial_error_message_count = error_message_count;

  /* Open the configuration file.  */
  fp = fopen (fname, "r");
  if (fp == NULL)
    return -1;

  /* The stream is not used by more than one thread.  */
  (void) __fsetlocking (fp, FSETLOCKING_BYCALLER);

  line = NULL;
  len = 0;

  do
    {
      ssize_t n = getline (&line, &len, fp);
      if (n < 0)
	break;
      if (line[n - 1] == '\n')
	line[n - 1] = '\0';

      /* Because the file format does not know any form of quoting we
	 can search forward for the next '#' character and if found
	 make it terminating the line.  */
      *strchrnul (line, '#') = '\0';

      /* If the line is blank it is ignored.  */
      if (line[0] == '\0')
	continue;

      entry = line;
      while (isspace (*entry) && *entry != '\0')
	++entry;
      cp = entry;
      while (!isspace (*cp) && *cp != '\0')
	++cp;
      arg1 = cp;
      ++arg1;
      *cp = '\0';
      if (strlen (entry) == 0)
	error (0, 0, _("Parse error: %s"), line);
      while (isspace (*arg1) && *arg1 != '\0')
	++arg1;
      cp = arg1;
      while (!isspace (*cp) && *cp != '\0')
	++cp;
      arg2 = cp;
      ++arg2;
      *cp = '\0';
      if (strlen (arg2) > 0)
	{
	  while (isspace (*arg2) && *arg2 != '\0')
	    ++arg2;
	  cp = arg2;
	  while (!isspace (*cp) && *cp != '\0')
	    ++cp;
	  *cp = '\0';
	}

      if (strcmp (entry, "positive-time-to-live") == 0)
	{
	  int idx = find_db (arg1);
	  if (idx >= 0)
	    dbs[idx].postimeout = atol (arg2);
	}
      else if (strcmp (entry, "negative-time-to-live") == 0)
	{
	  int idx = find_db (arg1);
	  if (idx >= 0)
	    dbs[idx].negtimeout = atol (arg2);
	}
      else if (strcmp (entry, "suggested-size") == 0)
	{
	  int idx = find_db (arg1);
	  if (idx >= 0)
	    dbs[idx].suggested_module
	      = atol (arg2) ?: DEFAULT_SUGGESTED_MODULE;
	}
      else if (strcmp (entry, "enable-cache") == 0)
	{
	  int idx = find_db (arg1);
	  if (idx >= 0)
	    {
	      if (strcmp (arg2, "no") == 0)
		dbs[idx].enabled = 0;
	      else if (strcmp (arg2, "yes") == 0)
		dbs[idx].enabled = 1;
	    }
	}
      else if (strcmp (entry, "check-files") == 0)
	{
	  int idx = find_db (arg1);
	  if (idx >= 0)
	    {
	      if (strcmp (arg2, "no") == 0)
		dbs[idx].check_file = 0;
	      else if (strcmp (arg2, "yes") == 0)
		dbs[idx].check_file = 1;
	    }
	}
      else if (strcmp (entry, "max-db-size") == 0)
	{
	  int idx = find_db (arg1);
	  if (idx >= 0)
	    dbs[idx].max_db_size = atol (arg2) ?: DEFAULT_MAX_DB_SIZE;
	}
      else if (strcmp (entry, "logfile") == 0)
	set_logfile (arg1);
      else if (strcmp (entry, "debug-level") == 0)
	{
	  int level = atoi (arg1);
	  if (level > 0)
	    debug_level = level;
	}
      else if (strcmp (entry, "threads") == 0)
	{
	  if (nthreads == -1)
	    nthreads = MAX (atol (arg1), lastdb);
	}
      else if (strcmp (entry, "max-threads") == 0)
	{
	  max_nthreads = MAX (atol (arg1), lastdb);
	}
      else if (strcmp (entry, "server-user") == 0)
	{
	  if (!arg1)
	    error (0, 0, _("Must specify user name for server-user option"));
	  else
	    {
	      free ((char *) server_user);
	      server_user = xstrdup (arg1);
	    }
	}
      else if (strcmp (entry, "stat-user") == 0)
	{
	  if (arg1 == NULL)
	    error (0, 0, _("Must specify user name for stat-user option"));
	  else
	    {
	      free ((char *) stat_user);
	      stat_user = xstrdup (arg1);

	      struct passwd *pw = getpwnam (stat_user);
	      if (pw != NULL)
		stat_uid = pw->pw_uid;
	    }
	}
      else if (strcmp (entry, "persistent") == 0)
	{
	  int idx = find_db (arg1);
	  if (idx >= 0)
	    {
	      if (strcmp (arg2, "no") == 0)
		dbs[idx].persistent = 0;
	      else if (strcmp (arg2, "yes") == 0)
		dbs[idx].persistent = 1;
	    }
	}
      else if (strcmp (entry, "shared") == 0)
	{
	  int idx = find_db (arg1);
	  if (idx >= 0)
	    {
	      if (strcmp (arg2, "no") == 0)
		dbs[idx].shared = 0;
	      else if (strcmp (arg2, "yes") == 0)
		dbs[idx].shared = 1;
	    }
	}
      else if (strcmp (entry, "reload-count") == 0)
	{
	  if (strcasecmp (arg1, "unlimited") == 0)
	    reload_count = UINT_MAX;
	  else
	    {
	      unsigned long int count = strtoul (arg1, NULL, 0);
	      if (count > UINT8_MAX - 1)
		reload_count = UINT_MAX;
	      else
		reload_count = count;
	    }
	}
      else if (strcmp (entry, "paranoia") == 0)
	{
	  if (strcmp (arg1, "no") == 0)
	    paranoia = 0;
	  else if (strcmp (arg1, "yes") == 0)
	    paranoia = 1;
	}
      else if (strcmp (entry, "restart-interval") == 0)
	{
	  if (arg1 != NULL)
	    restart_interval = atol (arg1);
	  else
	    error (0, 0, _("Must specify value for restart-interval option"));
	}
      else if (strcmp (entry, "auto-propagate") == 0)
	{
	  int idx = find_db (arg1);
	  if (idx >= 0)
	    {
	      if (strcmp (arg2, "no") == 0)
		dbs[idx].propagate = 0;
	      else if (strcmp (arg2, "yes") == 0)
		dbs[idx].propagate = 1;
	    }
	}
      else
	error (0, 0, _("Unknown option: %s %s %s"), entry, arg1, arg2);
    }
  while (!feof_unlocked (fp));

  if (paranoia)
    {
      restart_time = time (NULL) + restart_interval;

      /* Save the old current workding directory if we are in paranoia
	 mode.  We have to change back to it.  */
      oldcwd = get_current_dir_name ();
      if (oldcwd == NULL)
	{
	  error (0, 0, _("\
cannot get current working directory: %s; disabling paranoia mode"),
		   strerror (errno));
	  paranoia = 0;
	}
    }

  /* Enforce sanity.  */
  if (max_nthreads < nthreads)
    max_nthreads = nthreads;

  for (cnt = 0; cnt < lastdb; ++cnt)
    {
      size_t datasize = (sizeof (struct database_pers_head)
			 + roundup (dbs[cnt].suggested_module
				    * sizeof (ref_t), ALIGN)
			 + (dbs[cnt].suggested_module
			    * DEFAULT_DATASIZE_PER_BUCKET));
      if (datasize > dbs[cnt].max_db_size)
	{
	  error (0, 0, _("maximum file size for %s database too small"),
		   dbnames[cnt]);
	  dbs[cnt].max_db_size = datasize;
	}

    }

  /* Free the buffer.  */
  free (line);
  /* Close configuration file.  */
  fclose (fp);

  return error_message_count != initial_error_message_count;
}
