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

#include <string.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

#include "nis_xdr.h"
#include "nis_intern.h"

nis_server **
nis_getservlist (const_nis_name dir)
{
  nis_result *res;
  nis_server **serv;

  res = nis_lookup (dir, FOLLOW_LINKS);

  if (res != NULL && NIS_RES_STATUS (res) == NIS_SUCCESS)
    {
      unsigned long i;
      nis_server *server;

      serv =
	malloc (sizeof (nis_server *) *
		(NIS_RES_OBJECT (res)->DI_data.do_servers.do_servers_len + 1));
      if (__glibc_unlikely (serv == NULL))
	{
	  nis_freeresult (res);
	  return NULL;
	}

      for (i = 0; i < NIS_RES_OBJECT (res)->DI_data.do_servers.do_servers_len;
	   ++i)
	{
	  server =
	    &NIS_RES_OBJECT (res)->DI_data.do_servers.do_servers_val[i];
	  serv[i] = calloc (1, sizeof (nis_server));
	  if (__glibc_unlikely (serv[i] == NULL))
	    {
	    free_all:
	      while (i-- > 0)
		{
		  free (serv[i]->pkey.n_bytes);
		  if (serv[i]->ep.ep_val != NULL)
		    {
		      unsigned long int j;
		      for (j = 0; j < serv[i]->ep.ep_len; ++j)
			{
			  free (serv[i]->ep.ep_val[j].proto);
			  free (serv[i]->ep.ep_val[j].family);
			  free (serv[i]->ep.ep_val[j].uaddr);
			}
		      free (serv[i]->ep.ep_val);
		    }
		  free (serv[i]->name);
		  free (serv[i]);
		}

	      free (serv);

	      nis_freeresult (res);

	      return NULL;
	    }

	  if (server->name != NULL)
	    {
	      serv[i]->name = strdup (server->name);
	      if (__glibc_unlikely (serv[i]->name == NULL))
		{
		  ++i;
		  goto free_all;
		}
	    }

          serv[i]->ep.ep_len = server->ep.ep_len;
          if (serv[i]->ep.ep_len > 0)
            {
              unsigned long int j;

              serv[i]->ep.ep_val =
		malloc (server->ep.ep_len * sizeof (endpoint));
	      if (__glibc_unlikely (serv[i]->ep.ep_val == NULL))
		{
		  ++i;
		  goto free_all;
		}

              for (j = 0; j < serv[i]->ep.ep_len; ++j)
                {
                  if (server->ep.ep_val[j].uaddr)
                    serv[i]->ep.ep_val[j].uaddr =
		      strdup (server->ep.ep_val[j].uaddr);
                  else
                    serv[i]->ep.ep_val[j].uaddr = NULL;
                  if (server->ep.ep_val[j].family)
		    serv[i]->ep.ep_val[j].family =
		      strdup (server->ep.ep_val[j].family);
                  else
                    serv[i]->ep.ep_val[j].family = NULL;
                  if (server->ep.ep_val[j].proto)
		    serv[i]->ep.ep_val[j].proto =
		      strdup (server->ep.ep_val[j].proto);
                  else
		    serv[i]->ep.ep_val[j].proto = NULL;
                }
            }

          serv[i]->key_type = server->key_type;
          serv[i]->pkey.n_len = server->pkey.n_len;
          if (server->pkey.n_len > 0)
            {
              serv[i]->pkey.n_bytes = malloc (server->pkey.n_len);
              if (__glibc_unlikely (serv[i]->pkey.n_bytes == NULL))
		{
		  ++i;
		  goto free_all;
		}
              memcpy (serv[i]->pkey.n_bytes, server->pkey.n_bytes,
                      server->pkey.n_len);
            }
        }
      serv[i] = NULL;
    }
  else
    {
      serv = malloc (sizeof (nis_server *));
      if (__glibc_unlikely (serv != NULL))
	serv[0] = NULL;
    }

  nis_freeresult (res);

  return serv;
}
libnsl_hidden_nolink_def (nis_getservlist, GLIBC_2_1)

void
nis_freeservlist (nis_server **serv)
{
  int i;

  if (serv == NULL)
    return;

  i = 0;
  while (serv[i] != NULL)
    {
      xdr_free ((xdrproc_t)_xdr_nis_server, (char *)serv[i]);
      free (serv[i]);
      ++i;
    }
  free (serv);
}
libnsl_hidden_nolink_def (nis_freeservlist, GLIBC_2_1)
