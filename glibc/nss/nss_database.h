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

#ifndef _NSS_DATABASE_H
#define _NSS_DATABASE_H

#include <file_change_detection.h>

/* Each "line" in nsswitch.conf maps a supported database (example:
   passwd) to one or more name service providers (example: files dns).
   Internally, each name service provider (example: dns) is a
   dynamically loadable module (i.e. libnss_dns.so), managed by
   nss_module.h.  The sequence of providers and rules (example: files
   [SUCCESS=RETURN] dns) is mapped by nss_action.h to a cached entry
   which encodes the sequence of modules and rules.  Keeping track of
   all supported databases and their corresponding actions is done
   here.

   The key entry is __nss_database_get, which provides a set of
   actions which can be used with nss_lookup_function() and
   nss_next().  Callers should assume that these functions are fast,
   and should not cache the result longer than needed.  */

#include "nss_action.h"

/* The enumeration literal in enum nss_database for the database NAME
   (e.g., nss_database_hosts for hosts).  */
#define NSS_DATABASE_LITERAL(name) nss_database_##name

enum nss_database
{
#define DEFINE_DATABASE(name) NSS_DATABASE_LITERAL (name),
#include "databases.def"
#undef DEFINE_DATABASE

  /* Total number of databases.  */
  NSS_DATABASE_COUNT
};

/* Looks up the action list for DB and stores it in *ACTIONS.  Returns
   true on success or false on failure.  Success can mean that
   *ACTIONS is NULL.  */
bool __nss_database_get (enum nss_database db, nss_action_list *actions);
libc_hidden_proto (__nss_database_get)

/* Like __nss_database_get, but does not reload /etc/nsswitch.conf
   from disk.  This assumes that there has been a previous successful
   __nss_database_get call (which may not have returned any data).  */
nss_action_list __nss_database_get_noreload (enum nss_database db)
  attribute_hidden;

/* Called from __libc_freeres.  */
void __nss_database_freeres (void) attribute_hidden;

/* Internal type.  Exposed only for fork handling purposes.  */
struct nss_database_data
{
  struct file_change_detection nsswitch_conf;
  nss_action_list services[NSS_DATABASE_COUNT];
  int reload_disabled;          /* Actually bool; int for atomic access.  */
  bool initialized;
};

/* Called by fork in the parent process, before forking.  */
void __nss_database_fork_prepare_parent (struct nss_database_data *data)
  attribute_hidden;

/* Called by fork in the new subprocess, after forking.  */
void __nss_database_fork_subprocess (struct nss_database_data *data)
  attribute_hidden;

#endif /* _NSS_DATABASE_H */
