/* Mapping NSS services to action lists.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include "nss_database.h"

#include <allocate_once.h>
#include <array_length.h>
#include <assert.h>
#include <atomic.h>
#include <ctype.h>
#include <file_change_detection.h>
#include <libc-lock.h>
#include <netdb.h>
#include <stdio_ext.h>
#include <string.h>

struct nss_database_state
{
  struct nss_database_data data;
  __libc_lock_define (, lock);
  /* If "/" changes, we switched into a container and do NOT want to
     reload anything.  This data must be persistent across
     reloads.  */
  ino64_t root_ino;
  dev_t root_dev;
};


/* Global NSS database state.  Underlying type is "struct
   nss_database_state *" but the allocate_once API requires
   "void *".  */
static void *global_database_state;

/* Allocate and return pointer to nss_database_state object or
   on failure return NULL.  */
static void *
global_state_allocate (void *closure)
{
  struct nss_database_state *result =  malloc (sizeof (*result));
  if (result != NULL)
    {
      result->data.nsswitch_conf.size = -1; /* Force reload.  */
      memset (result->data.services, 0, sizeof (result->data.services));
      result->data.initialized = true;
      result->data.reload_disabled = false;
      __libc_lock_init (result->lock);
      result->root_ino = 0;
      result->root_dev = 0;
    }
  return result;
}

/* Return pointer to global NSS database state, allocating as
   required, or returning NULL on failure.  */
static struct nss_database_state *
nss_database_state_get (void)
{
  return allocate_once (&global_database_state, global_state_allocate,
			NULL, NULL);
}

/* Database default selections.  nis/compat mappings get turned into
   "files" for !LINK_OBSOLETE_NSL configurations.  */
enum nss_database_default
{
 nss_database_default_defconfig = 0, /* "nis [NOTFOUND=return] files".  */
 nss_database_default_compat, /* "compat [NOTFOUND=return] files".  */
 nss_database_default_dns,    /* "dns [!UNAVAIL=return] files".  */
 nss_database_default_files,    /* "files".  */
 nss_database_default_nis,    /* "nis".  */
 nss_database_default_nis_nisplus,    /* "nis nisplus".  */
 nss_database_default_none,      /* Empty list.  */

 NSS_DATABASE_DEFAULT_COUNT     /* Number of defaults.  */
};

/* Databases not listed default to nss_database_default_defconfig.  */
static const char per_database_defaults[NSS_DATABASE_COUNT] =
  {
   [nss_database_group] = nss_database_default_compat,
   [nss_database_group_compat] = nss_database_default_nis,
   [nss_database_gshadow] = nss_database_default_files,
   [nss_database_hosts] = nss_database_default_dns,
   [nss_database_initgroups] = nss_database_default_none,
   [nss_database_networks] = nss_database_default_dns,
   [nss_database_passwd] = nss_database_default_compat,
   [nss_database_passwd_compat] = nss_database_default_nis,
   [nss_database_publickey] = nss_database_default_nis_nisplus,
   [nss_database_shadow] = nss_database_default_compat,
   [nss_database_shadow_compat] = nss_database_default_nis,
  };

struct nss_database_default_cache
{
  nss_action_list caches[NSS_DATABASE_DEFAULT_COUNT];
};

static bool
nss_database_select_default (struct nss_database_default_cache *cache,
                             enum nss_database db, nss_action_list *result)
{
  enum nss_database_default def = per_database_defaults[db];
  *result = cache->caches[def];
  if (*result != NULL)
    return true;

  /* Determine the default line string.  */
  const char *line;
  switch (def)
    {
#ifdef LINK_OBSOLETE_NSL
    case nss_database_default_defconfig:
      line = "nis [NOTFOUND=return] files";
      break;
    case nss_database_default_compat:
      line =  "compat [NOTFOUND=return] files";
      break;
#endif

    case nss_database_default_dns:
      line = "dns [!UNAVAIL=return] files";
      break;

    case nss_database_default_files:
#ifndef LINK_OBSOLETE_NSL
    case nss_database_default_defconfig:
    case nss_database_default_compat:
#endif
      line = "files";
      break;

    case nss_database_default_nis:
      line = "nis";
      break;

    case nss_database_default_nis_nisplus:
      line = "nis nisplus";
      break;

    case nss_database_default_none:
      /* Very special case: Leave *result as NULL.  */
      return true;

    case NSS_DATABASE_DEFAULT_COUNT:
      __builtin_unreachable ();
    }
  if (def < 0 || def >= NSS_DATABASE_DEFAULT_COUNT)
    /* Tell GCC that line is initialized.  */
    __builtin_unreachable ();

  *result = __nss_action_parse (line);
  if (*result == NULL)
    {
      assert (errno == ENOMEM);
      return false;
    }
  return true;
}

/* database_name must be large enough for each individual name plus a
   null terminator.  */
typedef char database_name[14];
#define DEFINE_DATABASE(name) \
  _Static_assert (sizeof (#name) <= sizeof (database_name), #name);
#include "databases.def"
#undef DEFINE_DATABASE

static const database_name nss_database_name_array[] =
  {
#define DEFINE_DATABASE(name) #name,
#include "databases.def"
#undef DEFINE_DATABASE
  };

static int
name_search (const void *left, const void *right)
{
  return strcmp (left, right);
}

static int
name_to_database_index (const char *name)
{
  database_name *name_entry = bsearch (name, nss_database_name_array,
                                       array_length (nss_database_name_array),
                                       sizeof (database_name), name_search);
  if (name_entry == NULL)
    return -1;
  return name_entry - nss_database_name_array;
}

static bool
process_line (struct nss_database_data *data, char *line)
{
  /* Ignore leading white spaces.  ATTENTION: this is different from
     what is implemented in Solaris.  The Solaris man page says a line
     beginning with a white space character is ignored.  We regard
     this as just another misfeature in Solaris.  */
  while (isspace (line[0]))
    ++line;

  /* Recognize `<database> ":"'.  */
  char *name = line;
  while (line[0] != '\0' && !isspace (line[0]) && line[0] != ':')
    ++line;
  if (line[0] == '\0' || name == line)
    /* Syntax error.  Skip this line.  */
    return true;
  while (line[0] != '\0' && (isspace (line[0]) || line[0] == ':'))
    *line++ = '\0';

  int db = name_to_database_index (name);
  if (db < 0)
    /* Not our database e.g. sudoers, automount, etc.  */
    return true;

  nss_action_list result = __nss_action_parse (line);
  if (result == NULL)
    return false;
  data->services[db] = result;
  return true;
}

int
__nss_configure_lookup (const char *dbname, const char *service_line)
{
  int db;
  nss_action_list result;
  struct nss_database_state *local;

  /* Convert named database to index.  */
  db = name_to_database_index (dbname);
  if (db < 0)
    /* Not our database (e.g., sudoers).  */
    return -1;

  /* Force any load/cache/read whatever to happen, so we can override
     it.  */
  __nss_database_get (db, &result);

  local = nss_database_state_get ();

  result = __nss_action_parse (service_line);
  if (result == NULL)
    return -1;

  atomic_store_release (&local->data.reload_disabled, 1);
  local->data.services[db] = result;

#ifdef USE_NSCD
  __nss_database_custom[db] = true;
#endif

  return 0;
}

/* Iterate over the lines in FP, parse them, and store them in DATA.
   Return false on memory allocation failure, true on success.  */
static bool
nss_database_reload_1 (struct nss_database_data *data, FILE *fp)
{
  char *line = NULL;
  size_t line_allocated = 0;
  bool result = false;

  while (true)
    {
      ssize_t ret = __getline (&line, &line_allocated, fp);
      if (__ferror_unlocked (fp))
        break;
      if (__feof_unlocked (fp))
        {
          result = true;
          break;
        }
      assert (ret > 0);
      (void) ret;               /* For NDEBUG builds.  */

      if (!process_line (data, line))
        break;
    }

  free (line);
  return result;
}

static bool
nss_database_reload (struct nss_database_data *staging,
                     struct file_change_detection *initial)
{
  FILE *fp = fopen (_PATH_NSSWITCH_CONF, "rce");
  if (fp == NULL)
    switch (errno)
      {
      case EACCES:
      case EISDIR:
      case ELOOP:
      case ENOENT:
      case ENOTDIR:
      case EPERM:
        /* Ignore these errors.  They are persistent errors caused
           by file system contents.  */
        break;
      default:
        /* Other errors refer to resource allocation problems and
           need to be handled by the application.  */
        return false;
      }
  else
    /* No other threads have access to fp.  */
    __fsetlocking (fp, FSETLOCKING_BYCALLER);

  /* We start with all of *staging pointing to NULL.  */

  bool ok = true;
  if (fp != NULL)
    ok = nss_database_reload_1 (staging, fp);

  /* Now we have non-NULL entries where the user explictly listed the
     service in nsswitch.conf.  */

  /* Apply defaults.  */
  if (ok)
    {
      struct nss_database_default_cache cache = { };

      /* These three default to other services if the user listed the
	 other service.  */

      /* "shadow_compat" defaults to "passwd_compat" if only the
	 latter is given.  */
      if (staging->services[nss_database_shadow_compat] == NULL)
	staging->services[nss_database_shadow_compat] =
	  staging->services[nss_database_passwd_compat];

      /* "shadow" defaults to "passwd" if only the latter is
	 given.  */
      if (staging->services[nss_database_shadow] == NULL)
	staging->services[nss_database_shadow] =
	  staging->services[nss_database_passwd];

      /* "gshadow" defaults to "group" if only the latter is
	 given.  */
      if (staging->services[nss_database_gshadow] == NULL)
	staging->services[nss_database_gshadow] =
	  staging->services[nss_database_group];

      /* For anything still unspecified, load the default configs.  */

      for (int i = 0; i < NSS_DATABASE_COUNT; ++i)
        if (staging->services[i] == NULL)
          {
            ok = nss_database_select_default (&cache, i,
                                              &staging->services[i]);
            if (!ok)
              break;
          }
    }

  if (ok)
    ok = __file_change_detection_for_fp (&staging->nsswitch_conf, fp);

  if (fp != NULL)
    {
      int saved_errno = errno;
      fclose (fp);
      __set_errno (saved_errno);
    }

  if (ok && !__file_is_unchanged (&staging->nsswitch_conf, initial))
    /* Reload is required because the file changed while reading.  */
    staging->nsswitch_conf.size = -1;

  return ok;
}

static bool
nss_database_check_reload_and_get (struct nss_database_state *local,
                                   nss_action_list *result,
                                   enum nss_database database_index)
{
  struct __stat64_t64 str;

  /* Acquire MO is needed because the thread that sets reload_disabled
     may have loaded the configuration first, so synchronize with the
     Release MO store there.  */
  if (atomic_load_acquire (&local->data.reload_disabled))
    {
      *result = local->data.services[database_index];
      /* No reload, so there is no error.  */
      return true;
    }

  struct file_change_detection initial;
  if (!__file_change_detection_for_path (&initial, _PATH_NSSWITCH_CONF))
    return false;

  __libc_lock_lock (local->lock);
  if (__file_is_unchanged (&initial, &local->data.nsswitch_conf))
    {
      /* Configuration is up-to-date.  Read it and return it to the
         caller.  */
      *result = local->data.services[database_index];
      __libc_lock_unlock (local->lock);
      return true;
    }

  /* Before we reload, verify that "/" hasn't changed.  We assume that
     errors here are very unlikely, but the chance that we're entering
     a container is also very unlikely, so we err on the side of both
     very unlikely things not happening at the same time.  */
  if (__stat64_time64 ("/", &str) != 0
      || (local->root_ino != 0
	  && (str.st_ino != local->root_ino
	      ||  str.st_dev != local->root_dev)))
    {
      /* Change detected; disable reloading and return current state.  */
      atomic_store_release (&local->data.reload_disabled, 1);
      *result = local->data.services[database_index];
      __libc_lock_unlock (local->lock);
      return true;
    }
  local->root_ino = str.st_ino;
  local->root_dev = str.st_dev;
  __libc_lock_unlock (local->lock);

  /* Avoid overwriting the global configuration until we have loaded
     everything successfully.  Otherwise, if the file change
     information changes back to what is in the global configuration,
     the lookups would use the partially-written  configuration.  */
  struct nss_database_data staging = { .initialized = true, };

  bool ok = nss_database_reload (&staging, &initial);

  if (ok)
    {
      __libc_lock_lock (local->lock);

      /* See above for memory order.  */
      if (!atomic_load_acquire (&local->data.reload_disabled))
        /* This may go back in time if another thread beats this
           thread with the update, but in this case, a reload happens
           on the next NSS call.  */
        local->data = staging;

      *result = local->data.services[database_index];
      __libc_lock_unlock (local->lock);
    }

  return ok;
}

bool
__nss_database_get (enum nss_database db, nss_action_list *actions)
{
  struct nss_database_state *local = nss_database_state_get ();
  return nss_database_check_reload_and_get (local, actions, db);
}
libc_hidden_def (__nss_database_get)

nss_action_list
__nss_database_get_noreload (enum nss_database db)
{
  /* There must have been a previous __nss_database_get call.  */
  struct nss_database_state *local = atomic_load_acquire (&global_database_state);
  assert (local != NULL);

  __libc_lock_lock (local->lock);
  nss_action_list result = local->data.services[db];
  __libc_lock_unlock (local->lock);
  return result;
}

void __libc_freeres_fn_section
__nss_database_freeres (void)
{
  free (global_database_state);
  global_database_state = NULL;
}

void
__nss_database_fork_prepare_parent (struct nss_database_data *data)
{
  /* Do not use allocate_once to trigger loading unnecessarily.  */
  struct nss_database_state *local = atomic_load_acquire (&global_database_state);
  if (local == NULL)
    data->initialized = false;
  else
    {
      /* Make a copy of the configuration.  This approach was chosen
         because it avoids acquiring the lock during the actual
         fork.  */
      __libc_lock_lock (local->lock);
      *data = local->data;
      __libc_lock_unlock (local->lock);
    }
}

void
__nss_database_fork_subprocess (struct nss_database_data *data)
{
  struct nss_database_state *local = atomic_load_acquire (&global_database_state);
  if (data->initialized)
    {
      /* Restore the state at the point of the fork.  */
      assert (local != NULL);
      local->data = *data;
      __libc_lock_init (local->lock);
    }
  else if (local != NULL)
    /* The NSS configuration was loaded concurrently during fork.  We
       do not know its state, so we need to discard it.  */
    global_database_state = NULL;
}
