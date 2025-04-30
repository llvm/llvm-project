/* Parse a service line from nsswitch.conf.
   Copyright (c) 1996-2021 Free Software Foundation, Inc.
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

#include <nsswitch.h>

#include <ctype.h>
#include <string.h>
#include <stdbool.h>

/* Staging area during parsing.  */
#define DYNARRAY_STRUCT action_list
#define DYNARRAY_ELEMENT struct nss_action
#define DYNARRAY_PREFIX action_list_
#include <malloc/dynarray-skeleton.c>

/* Skip whitespace in line[].  */
#define SKIP_WS() \
  while (line[0] != '\0' && isspace (line[0]))	\
    ++line;

/* Read the source names:
        `( <source> ( "[" "!"? (<status> "=" <action> )+ "]" )? )*'
   */
static bool
nss_action_parse (const char *line, struct action_list *result)
{
  while (1)
    {
      SKIP_WS ();
      if (line[0] == '\0')
        /* No more sources specified.  */
        return true;

      /* Read <source> identifier.  */
      const char *name = line;
      while (line[0] != '\0' && !isspace (line[0]) && line[0] != '[')
        ++line;
      if (name == line)
        return true;

      struct nss_action new_service
        = { .module = __nss_module_allocate (name, line - name), };
      if (new_service.module == NULL)
        {
          /* Memory allocation error.  */
          action_list_mark_failed (result);
          return false;
        }
      nss_action_set_all (&new_service, NSS_ACTION_CONTINUE);
      nss_action_set (&new_service, NSS_STATUS_SUCCESS, NSS_ACTION_RETURN);
      nss_action_set (&new_service, NSS_STATUS_RETURN, NSS_ACTION_RETURN);

      SKIP_WS ();

      if (line[0] == '[')
        {
          /* Read criterions.  */

	  /* Skip the '['.  */
	  ++line;
	  SKIP_WS ();

          do
            {
              int not;
              enum nss_status status;
              lookup_actions action;

              /* Grok ! before name to mean all statuses but that one.  */
              not = line[0] == '!';
              if (not)
                ++line;

              /* Read status name.  */
              name = line;
              while (line[0] != '\0' && !isspace (line[0]) && line[0] != '='
                     && line[0] != ']')
                ++line;

              /* Compare with known statuses.  */
              if (line - name == 7)
                {
                  if (__strncasecmp (name, "SUCCESS", 7) == 0)
                    status = NSS_STATUS_SUCCESS;
                  else if (__strncasecmp (name, "UNAVAIL", 7) == 0)
                    status = NSS_STATUS_UNAVAIL;
                  else
                    return false;
                }
              else if (line - name == 8)
                {
                  if (__strncasecmp (name, "NOTFOUND", 8) == 0)
                    status = NSS_STATUS_NOTFOUND;
                  else if (__strncasecmp (name, "TRYAGAIN", 8) == 0)
                    status = NSS_STATUS_TRYAGAIN;
                  else
                    return false;
                }
              else
		return false;

	      SKIP_WS ();
              if (line[0] != '=')
                return false;

	      /* Skip the '='.  */
	      ++line;
	      SKIP_WS ();
              name = line;
              while (line[0] != '\0' && !isspace (line[0]) && line[0] != '='
                     && line[0] != ']')
                ++line;

              if (line - name == 6 && __strncasecmp (name, "RETURN", 6) == 0)
                action = NSS_ACTION_RETURN;
              else if (line - name == 8
                       && __strncasecmp (name, "CONTINUE", 8) == 0)
                action = NSS_ACTION_CONTINUE;
              else if (line - name == 5
                       && __strncasecmp (name, "MERGE", 5) == 0)
                action = NSS_ACTION_MERGE;
              else
                return false;

              if (not)
                {
                  /* Save the current action setting for this status,
                     set them all to the given action, and reset this one.  */
                  const lookup_actions save
                    = nss_action_get (&new_service, status);
                  nss_action_set_all (&new_service, action);
                  nss_action_set (&new_service, status, save);
                }
              else
                nss_action_set (&new_service, status, action);

	      SKIP_WS ();
            }
          while (line[0] != ']');

          /* Skip the ']'.  */
          ++line;
        }

      action_list_add (result, new_service);
    }
}

nss_action_list
 __nss_action_parse (const char *line)
{
  struct action_list list;
  action_list_init (&list);
  if (nss_action_parse (line, &list))
    {
      size_t size;
      struct nss_action null_service
        = { .module = NULL, };

      action_list_add (&list, null_service);
      size = action_list_size (&list);
      return __nss_action_allocate (action_list_begin (&list), size);
    }
  else if (action_list_has_failed (&list))
    {
      /* Memory allocation error.  */
      __set_errno (ENOMEM);
      return NULL;
    }
  else
    {
      /* Parse error.  */
      __set_errno (EINVAL);
      return NULL;
    }
}
