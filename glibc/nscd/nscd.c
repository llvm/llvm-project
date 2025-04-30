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

/* nscd - Name Service Cache Daemon. Caches passwd, group, and hosts.  */

#include <argp.h>
#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <libintl.h>
#include <locale.h>
#include <paths.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <stdarg.h>

#include "dbg_log.h"
#include "nscd.h"
#include "selinux.h"
#include "../nss/nsswitch.h"
#include <device-nrs.h>
#ifdef HAVE_INOTIFY
# include <sys/inotify.h>
#endif
#include <kernel-features.h>

/* Get libc version number.  */
#include <version.h>

#define PACKAGE _libc_intl_domainname

int do_shutdown;
int disabled_passwd;
int disabled_group;

typedef enum
{
  /* Running in background as daemon.  */
  RUN_DAEMONIZE,
  /* Running in foreground but otherwise behave like a daemon,
     i.e., detach from terminal and use syslog.  This allows
     better integration with services like systemd.  */
  RUN_FOREGROUND,
  /* Run in foreground in debug mode.  */
  RUN_DEBUG
} run_modes;

static run_modes run_mode = RUN_DAEMONIZE;

static const char *conffile = _PATH_NSCDCONF;

static const char *print_cache = NULL;

time_t start_time;

uintptr_t pagesize_m1;

int paranoia;
time_t restart_time;
time_t restart_interval = RESTART_INTERVAL;
const char *oldcwd;
uid_t old_uid;
gid_t old_gid;

static int check_pid (const char *file);
static int write_pid (const char *file);
static int monitor_child (int fd);

/* Name and version of program.  */
static void print_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *, struct argp_state *) = print_version;

/* Function to print some extra text in the help message.  */
static char *more_help (int key, const char *text, void *input);

/* Definitions of arguments for argp functions.  */
static const struct argp_option options[] =
{
  { "config-file", 'f', N_("NAME"), 0,
    N_("Read configuration data from NAME") },
  { "debug", 'd', NULL, 0,
    N_("Do not fork and display messages on the current tty") },
  { "print", 'p', N_("NAME"), 0,
    N_("Print contents of the offline cache file NAME") },
  { "foreground", 'F', NULL, 0,
    N_("Do not fork, but otherwise behave like a daemon") },
  { "nthreads", 't', N_("NUMBER"), 0, N_("Start NUMBER threads") },
  { "shutdown", 'K', NULL, 0, N_("Shut the server down") },
  { "statistics", 'g', NULL, 0, N_("Print current configuration statistics") },
  { "invalidate", 'i', N_("TABLE"), 0,
    N_("Invalidate the specified cache") },
  { "secure", 'S', N_("TABLE,yes"), OPTION_HIDDEN,
    N_("Use separate cache for each user")},
  { NULL, 0, NULL, 0, NULL }
};

/* Short description of program.  */
static const char doc[] = N_("Name Service Cache Daemon.");

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, NULL, doc, NULL, more_help
};

/* True if only statistics are requested.  */
static bool get_stats;
static int parent_fd = -1;

int
main (int argc, char **argv)
{
  int remaining;

  /* Set locale via LC_ALL.  */
  setlocale (LC_ALL, "");
  /* Set the text message domain.  */
  textdomain (PACKAGE);

  /* Determine if the kernel has SELinux support.  */
  nscd_selinux_enabled (&selinux_enabled);

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  if (remaining != argc)
    {
      error (0, 0, gettext ("wrong number of arguments"));
      argp_help (&argp, stdout, ARGP_HELP_SEE, program_invocation_short_name);
      exit (1);
    }

  /* Print the contents of the indicated cache file.  */
  if (print_cache != NULL)
    /* Does not return.  */
    nscd_print_cache (print_cache);

  /* Read the configuration file.  */
  if (nscd_parse_file (conffile, dbs) != 0)
    /* We couldn't read the configuration file.  We don't start the
       server.  */
    error (EXIT_FAILURE, 0,
	   _("failure while reading configuration file; this is fatal"));

  /* Do we only get statistics?  */
  if (get_stats)
    /* Does not return.  */
    receive_print_stats ();

  /* Check if we are already running. */
  if (check_pid (_PATH_NSCDPID))
    error (EXIT_FAILURE, 0, _("already running"));

  /* Remember when we started.  */
  start_time = time (NULL);

  /* Determine page size.  */
  pagesize_m1 = getpagesize () - 1;

  if (run_mode == RUN_DAEMONIZE || run_mode == RUN_FOREGROUND)
    {
      int i;
      pid_t pid;

      /* Behave like a daemon.  */
      if (run_mode == RUN_DAEMONIZE)
	{
	  int fd[2];

	  if (pipe (fd) != 0)
	    error (EXIT_FAILURE, errno,
		   _("cannot create a pipe to talk to the child"));

	  pid = fork ();
	  if (pid == -1)
	    error (EXIT_FAILURE, errno, _("cannot fork"));
	  if (pid != 0)
	    {
	      /* The parent only reads from the child.  */
	      close (fd[1]);
	      exit (monitor_child (fd[0]));
	    }
	  else
	    {
	      /* The child only writes to the parent.  */
	      close (fd[0]);
	      parent_fd = fd[1];
	    }
	}

      int nullfd = open (_PATH_DEVNULL, O_RDWR);
      if (nullfd != -1)
	{
	  struct stat64 st;

	  if (fstat64 (nullfd, &st) == 0 && S_ISCHR (st.st_mode) != 0
#if defined DEV_NULL_MAJOR && defined DEV_NULL_MINOR
	      && st.st_rdev == makedev (DEV_NULL_MAJOR, DEV_NULL_MINOR)
#endif
	      )
	    {
	      /* It is the /dev/null special device alright.  */
	      (void) dup2 (nullfd, STDIN_FILENO);
	      (void) dup2 (nullfd, STDOUT_FILENO);
	      (void) dup2 (nullfd, STDERR_FILENO);

	      if (nullfd > 2)
		close (nullfd);
	    }
	  else
	    {
	      /* Ugh, somebody is trying to play a trick on us.  */
	      close (nullfd);
	      nullfd = -1;
	    }
	}
      int min_close_fd = nullfd == -1 ? 0 : STDERR_FILENO + 1;

      DIR *d = opendir ("/proc/self/fd");
      if (d != NULL)
	{
	  struct dirent64 *dirent;
	  int dfdn = dirfd (d);

	  while ((dirent = readdir64 (d)) != NULL)
	    {
	      char *endp;
	      long int fdn = strtol (dirent->d_name, &endp, 10);

	      if (*endp == '\0' && fdn != dfdn && fdn >= min_close_fd
		  && fdn != parent_fd)
		close ((int) fdn);
	    }

	  closedir (d);
	}
      else
	for (i = min_close_fd; i < getdtablesize (); i++)
	  if (i != parent_fd)
	    close (i);

      setsid ();

      if (chdir ("/") != 0)
	do_exit (EXIT_FAILURE, errno,
		 _("cannot change current working directory to \"/\""));

      openlog ("nscd", LOG_CONS | LOG_ODELAY, LOG_DAEMON);

      if (write_pid (_PATH_NSCDPID) < 0)
	dbg_log ("%s: %s", _PATH_NSCDPID, strerror (errno));

      if (!init_logfile ())
	dbg_log (_("Could not create log file"));

      /* Ignore job control signals.  */
      signal (SIGTTOU, SIG_IGN);
      signal (SIGTTIN, SIG_IGN);
      signal (SIGTSTP, SIG_IGN);
    }
  else
    /* In debug mode we are not paranoid.  */
    paranoia = 0;

  signal (SIGINT, termination_handler);
  signal (SIGQUIT, termination_handler);
  signal (SIGTERM, termination_handler);
  signal (SIGPIPE, SIG_IGN);

  /* Cleanup files created by a previous 'bind'.  */
  unlink (_PATH_NSCDSOCKET);

#ifdef HAVE_INOTIFY
  /* Use inotify to recognize changed files.  */
  inotify_fd = inotify_init1 (IN_NONBLOCK);
# ifndef __ASSUME_IN_NONBLOCK
  if (inotify_fd == -1 && errno == ENOSYS)
    {
      inotify_fd = inotify_init ();
      if (inotify_fd != -1)
	fcntl (inotify_fd, F_SETFL, O_RDONLY | O_NONBLOCK);
    }
# endif
#endif

#ifdef USE_NSCD
  /* Make sure we do not get recursive calls.  */
  __nss_disable_nscd (register_traced_file);
#endif

  /* Init databases.  */
  nscd_init ();

  /* Start the SELinux AVC.  */
  if (selinux_enabled)
    nscd_avc_init ();

  /* Handle incoming requests */
  start_threads ();

  return 0;
}


static void __attribute__ ((noreturn))
invalidate_db (const char *dbname)
{
  int sock = nscd_open_socket ();

  if (sock == -1)
    exit (EXIT_FAILURE);

  size_t dbname_len = strlen (dbname) + 1;
  size_t reqlen = sizeof (request_header) + dbname_len;
  struct
  {
    request_header req;
    char dbname[];
  } *reqdata = alloca (reqlen);

  reqdata->req.key_len = dbname_len;
  reqdata->req.version = NSCD_VERSION;
  reqdata->req.type = INVALIDATE;
  memcpy (reqdata->dbname, dbname, dbname_len);

  ssize_t nbytes = TEMP_FAILURE_RETRY (send (sock, reqdata, reqlen,
					     MSG_NOSIGNAL));

  if (nbytes != reqlen)
    {
      int err = errno;
      close (sock);
      error (EXIT_FAILURE, err, _("write incomplete"));
    }

  /* Wait for ack.  Older nscd just closed the socket when
     prune_cache finished, silently ignore that.  */
  int32_t resp = 0;
  nbytes = TEMP_FAILURE_RETRY (read (sock, &resp, sizeof (resp)));
  if (nbytes != 0 && nbytes != sizeof (resp))
    {
      int err = errno;
      close (sock);
      error (EXIT_FAILURE, err, _("cannot read invalidate ACK"));
    }

  close (sock);

  if (resp != 0)
    error (EXIT_FAILURE, resp, _("invalidation failed"));

  exit (0);
}

static void __attribute__ ((noreturn))
send_shutdown (void)
{
  int sock = nscd_open_socket ();

  if (sock == -1)
    exit (EXIT_FAILURE);

  request_header req;
  req.version = NSCD_VERSION;
  req.type = SHUTDOWN;
  req.key_len = 0;

  ssize_t nbytes = TEMP_FAILURE_RETRY (send (sock, &req, sizeof req,
                                             MSG_NOSIGNAL));
  close (sock);
  exit (nbytes != sizeof (request_header) ? EXIT_FAILURE : EXIT_SUCCESS);
}

/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'd':
      ++debug_level;
      run_mode = RUN_DEBUG;
      break;

    case 'p':
      print_cache = arg;
      break;

    case 'F':
      run_mode = RUN_FOREGROUND;
      break;

    case 'f':
      conffile = arg;
      break;

    case 'K':
      if (getuid () != 0)
	error (4, 0, _("Only root is allowed to use this option!"));
      else
        send_shutdown ();
      break;

    case 'g':
      get_stats = true;
      break;

    case 'i':
      {
        /* Validate the database name.  */

        dbtype cnt;
        for (cnt = pwddb; cnt < lastdb; ++cnt)
          if (strcmp (arg, dbnames[cnt]) == 0)
            break;

        if (cnt == lastdb)
          {
            argp_error (state, _("'%s' is not a known database"), arg);
            return EINVAL;
          }
      }
      if (getuid () != 0)
	error (4, 0, _("Only root is allowed to use this option!"));
      else
        invalidate_db (arg);
      break;

    case 't':
      nthreads = atol (arg);
      break;

    case 'S':
      error (0, 0, _("secure services not implemented anymore"));
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }

  return 0;
}

/* Print bug-reporting information in the help message.  */
static char *
more_help (int key, const char *text, void *input)
{
  switch (key)
    {
    case ARGP_KEY_HELP_EXTRA:
      {
	/* We print some extra information.  */

	char *tables = xstrdup (dbnames[0]);
	for (dbtype i = 1; i < lastdb; ++i)
	  {
	    char *more_tables;
	    if (asprintf (&more_tables, "%s %s", tables, dbnames[i]) < 0)
	      more_tables = NULL;
	    free (tables);
	    if (more_tables == NULL)
	      return NULL;
	    tables = more_tables;
	  }

	char *tp;
	if (asprintf (&tp, gettext ("\
Supported tables:\n\
%s\n\
\n\
For bug reporting instructions, please see:\n\
%s.\n\
"), tables, REPORT_BUGS_TO) < 0)
	  tp = NULL;
	free (tables);
	return tp;
      }

    default:
      break;
    }

  return (char *) text;
}

/* Print the version information.  */
static void
print_version (FILE *stream, struct argp_state *state)
{
  fprintf (stream, "nscd %s%s\n", PKGVERSION, VERSION);
  fprintf (stream, gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
  fprintf (stream, gettext ("Written by %s.\n"),
	   "Thorsten Kukuk and Ulrich Drepper");
}


/* Create a socket connected to a name.  */
int
nscd_open_socket (void)
{
  struct sockaddr_un addr;
  int sock;

  sock = socket (PF_UNIX, SOCK_STREAM, 0);
  if (sock < 0)
    return -1;

  addr.sun_family = AF_UNIX;
  assert (sizeof (addr.sun_path) >= sizeof (_PATH_NSCDSOCKET));
  strcpy (addr.sun_path, _PATH_NSCDSOCKET);
  if (connect (sock, (struct sockaddr *) &addr, sizeof (addr)) < 0)
    {
      close (sock);
      return -1;
    }

  return sock;
}


/* Cleanup.  */
void
termination_handler (int signum)
{
  close_sockets ();

  /* Clean up the file created by 'bind'.  */
  unlink (_PATH_NSCDSOCKET);

  /* Clean up pid file.  */
  unlink (_PATH_NSCDPID);

  // XXX Terminate threads.

  /* Synchronize memory.  */
  for (int cnt = 0; cnt < lastdb; ++cnt)
    {
      if (!dbs[cnt].enabled || dbs[cnt].head == NULL)
	continue;

      /* Make sure nobody keeps using the database.  */
      dbs[cnt].head->timestamp = 0;

      if (dbs[cnt].persistent)
	// XXX async OK?
	msync (dbs[cnt].head, dbs[cnt].memsize, MS_ASYNC);
    }

  _exit (EXIT_SUCCESS);
}

/* Returns 1 if the process in pid file FILE is running, 0 if not.  */
static int
check_pid (const char *file)
{
  FILE *fp;

  fp = fopen (file, "r");
  if (fp)
    {
      pid_t pid;
      int n;

      n = fscanf (fp, "%d", &pid);
      fclose (fp);

      /* If we cannot parse the file default to assuming nscd runs.
	 If the PID is alive, assume it is running.  That all unless
	 the PID is the same as the current process' since tha latter
	 can mean we re-exec.  */
      if ((n != 1 || kill (pid, 0) == 0) && pid != getpid ())
	return 1;
    }

  return 0;
}

/* Write the current process id to the file FILE.
   Returns 0 if successful, -1 if not.  */
static int
write_pid (const char *file)
{
  FILE *fp;

  fp = fopen (file, "w");
  if (fp == NULL)
    return -1;

  fprintf (fp, "%d\n", getpid ());

  int result = fflush (fp) || ferror (fp) ? -1 : 0;

  fclose (fp);

  return result;
}

static int
monitor_child (int fd)
{
  int child_ret = 0;
  int ret = read (fd, &child_ret, sizeof (child_ret));

  /* The child terminated with an error, either via exit or some other abnormal
     method, like a segfault.  */
  if (ret <= 0 || child_ret != 0)
    {
      int status;
      int err = wait (&status);

      if (err < 0)
	{
	  fprintf (stderr, _("'wait' failed\n"));
	  return 1;
	}

      if (WIFEXITED (status))
	{
	  child_ret = WEXITSTATUS (status);
	  fprintf (stderr, _("child exited with status %d\n"), child_ret);
	}
      if (WIFSIGNALED (status))
	{
	  child_ret = WTERMSIG (status);
	  fprintf (stderr, _("child terminated by signal %d\n"), child_ret);
	}
    }

  /* We have the child status, so exit with that code.  */
  close (fd);

  return child_ret;
}

void
do_exit (int child_ret, int errnum, const char *format, ...)
{
  if (parent_fd != -1)
    {
      int ret __attribute__ ((unused));
      ret = write (parent_fd, &child_ret, sizeof (child_ret));
      assert (ret == sizeof (child_ret));
      close (parent_fd);
    }

  if (format != NULL)
    {
      /* Emulate error() since we don't have a va_list variant for it.  */
      va_list argp;

      fflush (stdout);

      fprintf (stderr, "%s: ", program_invocation_name);

      va_start (argp, format);
      vfprintf (stderr, format, argp);
      va_end (argp);

      fprintf (stderr, ": %s\n", strerror (errnum));
      fflush (stderr);
    }

  /* Finally, exit.  */
  exit (child_ret);
}

void
notify_parent (int child_ret)
{
  if (parent_fd == -1)
    return;

  int ret __attribute__ ((unused));
  ret = write (parent_fd, &child_ret, sizeof (child_ret));
  assert (ret == sizeof (child_ret));
  close (parent_fd);
  parent_fd = -1;
}
