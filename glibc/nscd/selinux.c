/* SELinux access controls for nscd.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Matthew Rickard <mjricka@epoch.ncsc.mil>, 2004.

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

#include "config.h"
#include <error.h>
#include <errno.h>
#include <libintl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <syslog.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <selinux/avc.h>
#include <selinux/selinux.h>
#ifdef HAVE_LIBAUDIT
# include <libaudit.h>
#endif
#include <libc-diag.h>

#include "dbg_log.h"
#include "selinux.h"


#ifdef HAVE_SELINUX
/* Global variable to tell if the kernel has SELinux support.  */
int selinux_enabled;

/* Define mappings of request type to AVC permission name.  */
static const char *perms[LASTREQ] =
{
  [GETPWBYNAME] = "getpwd",
  [GETPWBYUID] = "getpwd",
  [GETGRBYNAME] = "getgrp",
  [GETGRBYGID] = "getgrp",
  [GETHOSTBYNAME] = "gethost",
  [GETHOSTBYNAMEv6] = "gethost",
  [GETHOSTBYADDR] = "gethost",
  [GETHOSTBYADDRv6] = "gethost",
  [SHUTDOWN] = "admin",
  [GETSTAT] = "getstat",
  [INVALIDATE] = "admin",
  [GETFDPW] = "shmempwd",
  [GETFDGR] = "shmemgrp",
  [GETFDHST] = "shmemhost",
  [GETAI] = "gethost",
  [INITGROUPS] = "getgrp",
  [GETSERVBYNAME] = "getserv",
  [GETSERVBYPORT] = "getserv",
  [GETFDSERV] = "shmemserv",
  [GETNETGRENT] = "getnetgrp",
  [INNETGR] = "getnetgrp",
  [GETFDNETGR] = "shmemnetgrp",
};

/* Store an entry ref to speed AVC decisions.  */
static struct avc_entry_ref aeref;

/* Thread to listen for SELinux status changes via netlink.  */
static pthread_t avc_notify_thread;

#ifdef HAVE_LIBAUDIT
/* Prototype for supporting the audit daemon */
static void log_callback (const char *fmt, ...);
#endif

/* Prototypes for AVC callback functions.  */
static void *avc_create_thread (void (*run) (void));
static void avc_stop_thread (void *thread);
static void *avc_alloc_lock (void);
static void avc_get_lock (void *lock);
static void avc_release_lock (void *lock);
static void avc_free_lock (void *lock);

/* AVC callback structures for use in avc_init.  */
static const struct avc_log_callback log_cb =
{
#ifdef HAVE_LIBAUDIT
  .func_log = log_callback,
#else
  .func_log = dbg_log,
#endif
  .func_audit = NULL
};
static const struct avc_thread_callback thread_cb =
{
  .func_create_thread = avc_create_thread,
  .func_stop_thread = avc_stop_thread
};
static const struct avc_lock_callback lock_cb =
{
  .func_alloc_lock = avc_alloc_lock,
  .func_get_lock = avc_get_lock,
  .func_release_lock = avc_release_lock,
  .func_free_lock = avc_free_lock
};

#ifdef HAVE_LIBAUDIT
/* The audit system's netlink socket descriptor */
static int audit_fd = -1;

/* When an avc denial occurs, log it to audit system */
static void
log_callback (const char *fmt, ...)
{
  if (audit_fd >= 0)
    {
      va_list ap;
      va_start (ap, fmt);

      char *buf;
      int e = vasprintf (&buf, fmt, ap);
      if (e < 0)
	{
	  buf = alloca (BUFSIZ);
	  vsnprintf (buf, BUFSIZ, fmt, ap);
	}

      /* FIXME: need to attribute this to real user, using getuid for now */
      audit_log_user_avc_message (audit_fd, AUDIT_USER_AVC, buf, NULL, NULL,
				  NULL, getuid ());

      if (e >= 0)
	free (buf);

      va_end (ap);
    }
}

/* Initialize the connection to the audit system */
static void
audit_init (void)
{
  audit_fd = audit_open ();
  if (audit_fd < 0
      /* If kernel doesn't support audit, bail out */
      && errno != EINVAL && errno != EPROTONOSUPPORT && errno != EAFNOSUPPORT)
    dbg_log (_("Failed opening connection to the audit subsystem: %m"));
}


# ifdef HAVE_LIBCAP
static const cap_value_t new_cap_list[] =
  { CAP_AUDIT_WRITE };
#  define nnew_cap_list (sizeof (new_cap_list) / sizeof (new_cap_list[0]))
static const cap_value_t tmp_cap_list[] =
  { CAP_AUDIT_WRITE, CAP_SETUID, CAP_SETGID };
#  define ntmp_cap_list (sizeof (tmp_cap_list) / sizeof (tmp_cap_list[0]))

cap_t
preserve_capabilities (void)
{
  if (getuid () != 0)
    /* Not root, then we cannot preserve anything.  */
    return NULL;

  if (prctl (PR_SET_KEEPCAPS, 1) == -1)
    {
      dbg_log (_("Failed to set keep-capabilities"));
      do_exit (EXIT_FAILURE, errno, _("prctl(KEEPCAPS) failed"));
      /* NOTREACHED */
    }

  cap_t tmp_caps = cap_init ();
  cap_t new_caps = NULL;
  if (tmp_caps != NULL)
    new_caps = cap_init ();

  if (tmp_caps == NULL || new_caps == NULL)
    {
      if (tmp_caps != NULL)
	cap_free (tmp_caps);

      dbg_log (_("Failed to initialize drop of capabilities"));
      do_exit (EXIT_FAILURE, 0, _("cap_init failed"));
    }

  /* There is no reason why these should not work.  */
  cap_set_flag (new_caps, CAP_PERMITTED, nnew_cap_list,
		(cap_value_t *) new_cap_list, CAP_SET);
  cap_set_flag (new_caps, CAP_EFFECTIVE, nnew_cap_list,
		(cap_value_t *) new_cap_list, CAP_SET);

  cap_set_flag (tmp_caps, CAP_PERMITTED, ntmp_cap_list,
		(cap_value_t *) tmp_cap_list, CAP_SET);
  cap_set_flag (tmp_caps, CAP_EFFECTIVE, ntmp_cap_list,
		(cap_value_t *) tmp_cap_list, CAP_SET);

  int res = cap_set_proc (tmp_caps);

  cap_free (tmp_caps);

  if (__glibc_unlikely (res != 0))
    {
      cap_free (new_caps);
      dbg_log (_("Failed to drop capabilities"));
      do_exit (EXIT_FAILURE, 0, _("cap_set_proc failed"));
    }

  return new_caps;
}

void
install_real_capabilities (cap_t new_caps)
{
  /* If we have no capabilities there is nothing to do here.  */
  if (new_caps == NULL)
    return;

  if (cap_set_proc (new_caps))
    {
      cap_free (new_caps);
      dbg_log (_("Failed to drop capabilities"));
      do_exit (EXIT_FAILURE, 0, _("cap_set_proc failed"));
      /* NOTREACHED */
    }

  cap_free (new_caps);

  if (prctl (PR_SET_KEEPCAPS, 0) == -1)
    {
      dbg_log (_("Failed to unset keep-capabilities"));
      do_exit (EXIT_FAILURE, errno, _("prctl(KEEPCAPS) failed"));
      /* NOTREACHED */
    }
}
# endif /* HAVE_LIBCAP */
#endif /* HAVE_LIBAUDIT */

/* Determine if we are running on an SELinux kernel. Set selinux_enabled
   to the result.  */
void
nscd_selinux_enabled (int *selinux_enabled)
{
  *selinux_enabled = is_selinux_enabled ();
  if (*selinux_enabled < 0)
    {
      dbg_log (_("Failed to determine if kernel supports SELinux"));
      do_exit (EXIT_FAILURE, 0, NULL);
    }
}


/* Create thread for AVC netlink notification.  */
static void *
avc_create_thread (void (*run) (void))
{
  int rc;

  rc =
    pthread_create (&avc_notify_thread, NULL, (void *(*) (void *)) run, NULL);
  if (rc != 0)
    do_exit (EXIT_FAILURE, rc, _("Failed to start AVC thread"));

  return &avc_notify_thread;
}


/* Stop AVC netlink thread.  */
static void
avc_stop_thread (void *thread)
{
  pthread_cancel (*(pthread_t *) thread);
}


/* Allocate a new AVC lock.  */
static void *
avc_alloc_lock (void)
{
  pthread_mutex_t *avc_mutex;

  avc_mutex = malloc (sizeof (pthread_mutex_t));
  if (avc_mutex == NULL)
    do_exit (EXIT_FAILURE, errno, _("Failed to create AVC lock"));
  pthread_mutex_init (avc_mutex, NULL);

  return avc_mutex;
}


/* Acquire an AVC lock.  */
static void
avc_get_lock (void *lock)
{
  pthread_mutex_lock (lock);
}


/* Release an AVC lock.  */
static void
avc_release_lock (void *lock)
{
  pthread_mutex_unlock (lock);
}


/* Free an AVC lock.  */
static void
avc_free_lock (void *lock)
{
  pthread_mutex_destroy (lock);
  free (lock);
}


/* avc_init (along with several other symbols) was marked as deprecated by the
   SELinux API starting from version 3.1.  We use it here, but should
   eventually switch to the newer API.  */
DIAG_PUSH_NEEDS_COMMENT
DIAG_IGNORE_NEEDS_COMMENT (10, "-Wdeprecated-declarations");

/* Initialize the user space access vector cache (AVC) for NSCD along with
   log/thread/lock callbacks.  */
void
nscd_avc_init (void)
{
  avc_entry_ref_init (&aeref);

  if (avc_init ("avc", NULL, &log_cb, &thread_cb, &lock_cb) < 0)
    do_exit (EXIT_FAILURE, errno, _("Failed to start AVC"));
  else
    dbg_log (_("Access Vector Cache (AVC) started"));
#ifdef HAVE_LIBAUDIT
  audit_init ();
#endif
}
DIAG_POP_NEEDS_COMMENT


/* security_context_t and sidput (along with several other symbols) were marked
   as deprecated by the SELinux API starting from version 3.1.  We use them
   here, but should eventually switch to the newer API.  */
DIAG_PUSH_NEEDS_COMMENT
DIAG_IGNORE_NEEDS_COMMENT (10, "-Wdeprecated-declarations");

/* Check the permission from the caller (via getpeercon) to nscd.
   Returns 0 if access is allowed, 1 if denied, and -1 on error.

   The SELinux policy, enablement, and permission bits are all dynamic and the
   caching done by glibc is not entirely correct.  This nscd support should be
   rewritten to use selinux_check_permission.  A rewrite is risky though and
   requires some refactoring.  Currently we use symbolic mappings instead of
   compile time constants (which SELinux upstream says are going away), and we
   use security_deny_unknown to determine what to do if selinux-policy* doesn't
   have a definition for the the permission or object class we are looking
   up.  */
int
nscd_request_avc_has_perm (int fd, request_type req)
{
  /* Initialize to NULL so we know what to free in case of failure.  */
  security_context_t scon = NULL;
  security_context_t tcon = NULL;
  security_id_t ssid = NULL;
  security_id_t tsid = NULL;
  int rc = -1;
  security_class_t sc_nscd;
  access_vector_t perm;
  int avc_deny_unknown;

  /* Check if SELinux denys or allows unknown object classes
     and permissions.  It is 0 if they are allowed, 1 if they
     are not allowed and -1 on error.  */
  if ((avc_deny_unknown = security_deny_unknown ()) == -1)
    dbg_log (_("Error querying policy for undefined object classes "
	       "or permissions."));

  /* Get the security class for nscd.  If this fails we will likely be
     unable to do anything unless avc_deny_unknown is 0.  */
  sc_nscd = string_to_security_class ("nscd");
  if (sc_nscd == 0 && avc_deny_unknown == 1)
    dbg_log (_("Error getting security class for nscd."));

  /* Convert permission to AVC bits.  */
  perm = string_to_av_perm (sc_nscd, perms[req]);
  if (perm == 0 && avc_deny_unknown == 1)
      dbg_log (_("Error translating permission name "
		 "\"%s\" to access vector bit."), perms[req]);

  /* If the nscd security class was not found or perms were not
     found and AVC does not deny unknown values then allow it.  */
  if ((sc_nscd == 0 || perm == 0) && avc_deny_unknown == 0)
    return 0;

  if (getpeercon (fd, &scon) < 0)
    {
      dbg_log (_("Error getting context of socket peer"));
      goto out;
    }
  if (getcon (&tcon) < 0)
    {
      dbg_log (_("Error getting context of nscd"));
      goto out;
    }
  if (avc_context_to_sid (scon, &ssid) < 0
      || avc_context_to_sid (tcon, &tsid) < 0)
    {
      dbg_log (_("Error getting sid from context"));
      goto out;
    }

  /* The SELinux API for avc_has_perm conflates access denied and error into
     the return code -1, while nscd_request_avs_has_perm has distinct error
     (-1) and denied (1) return codes. We map the avc_has_perm access denied or
     error into an access denied at the nscd interface level (we do accurately
     report error for the getpeercon, getcon, and avc_context_to_sid interfaces
     used above).  */
  rc = avc_has_perm (ssid, tsid, sc_nscd, perm, &aeref, NULL) < 0;

out:
  if (scon)
    freecon (scon);
  if (tcon)
    freecon (tcon);
  if (ssid)
    sidput (ssid);
  if (tsid)
    sidput (tsid);

  return rc;
}
DIAG_POP_NEEDS_COMMENT


/* Wrapper to get AVC statistics.  */
void
nscd_avc_cache_stats (struct avc_cache_stats *cstats)
{
  avc_cache_stats (cstats);
}


/* Print the AVC statistics to stdout.  */
void
nscd_avc_print_stats (struct avc_cache_stats *cstats)
{
  printf (_("\nSELinux AVC Statistics:\n\n"
	    "%15u  entry lookups\n"
	    "%15u  entry hits\n"
	    "%15u  entry misses\n"
	    "%15u  entry discards\n"
	    "%15u  CAV lookups\n"
	    "%15u  CAV hits\n"
	    "%15u  CAV probes\n"
	    "%15u  CAV misses\n"),
	  cstats->entry_lookups, cstats->entry_hits, cstats->entry_misses,
	  cstats->entry_discards, cstats->cav_lookups, cstats->cav_hits,
	  cstats->cav_probes, cstats->cav_misses);
}

#endif /* HAVE_SELINUX */
