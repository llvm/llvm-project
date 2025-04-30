/* Find network interface names and index numbers.  Hurd version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <error.h>
#include <net/if.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <hurd.h>
#include <hurd/ioctl.h>
#include <hurd/pfinet.h>

/* Return the interface index corresponding to interface IFNAME.
   On error, return 0.  */
unsigned int
__if_nametoindex (const char *ifname)
{
  struct ifreq ifr;
  int fd = __opensock ();

  if (fd < 0)
    return 0;

  if (strlen (ifname) >= IFNAMSIZ)
    {
      __set_errno (ENODEV);
      return 0;
    }

  strncpy (ifr.ifr_name, ifname, IFNAMSIZ);
  if (__ioctl (fd, SIOCGIFINDEX, &ifr) < 0)
    {
      int saved_errno = errno;
      __close (fd);
      if (saved_errno == EINVAL || saved_errno == ENOTTY)
        __set_errno (ENOSYS);
      return 0;
    }
  __close (fd);
  return ifr.ifr_ifindex;
}
libc_hidden_def (__if_nametoindex)
weak_alias (__if_nametoindex, if_nametoindex)
libc_hidden_weak (if_nametoindex)

/* Free the structure IFN returned by if_nameindex.  */
void
__if_freenameindex (struct if_nameindex *ifn)
{
  struct if_nameindex *ptr = ifn;
  while (ptr->if_name || ptr->if_index)
    {
      free (ptr->if_name);
      ++ptr;
    }
  free (ifn);
}
libc_hidden_def (__if_freenameindex)
weak_alias (__if_freenameindex, if_freenameindex)
libc_hidden_weak (if_freenameindex)

/* Return an array of if_nameindex structures, one for each network
   interface present, plus one indicating the end of the array.  On
   error, return NULL.  */
struct if_nameindex *
__if_nameindex (void)
{
  error_t err = 0;
  char data[2048];
  file_t server;
  int fd = __opensock ();
  struct ifconf ifc;
  unsigned int nifs, i;
  struct if_nameindex *idx = NULL;

  ifc.ifc_buf = data;

  if (fd < 0)
    return NULL;

  server = _hurd_socket_server (PF_INET, 0);
  if (server == MACH_PORT_NULL)
    nifs = 0;
  else
    {
      size_t len = sizeof data;
      err = __pfinet_siocgifconf (server, -1, &ifc.ifc_buf, &len);
      if (err == MACH_SEND_INVALID_DEST || err == MIG_SERVER_DIED)
	{
	  /* On the first use of the socket server during the operation,
	     allow for the old server port dying.  */
	  server = _hurd_socket_server (PF_INET, 1);
	  if (server == MACH_PORT_NULL)
	    goto out;
	  err = __pfinet_siocgifconf (server, -1, &ifc.ifc_buf, &len);
	}
      if (err)
	goto out;

      ifc.ifc_len = len;
      nifs = len / sizeof (struct ifreq);
    }

  idx = malloc ((nifs + 1) * sizeof (struct if_nameindex));
  if (idx == NULL)
    {
      err = ENOBUFS;
      goto out;
    }

  for (i = 0; i < nifs; ++i)
    {
      struct ifreq *ifr = &ifc.ifc_req[i];
      idx[i].if_name = __strdup (ifr->ifr_name);
      if (idx[i].if_name == NULL
          || __ioctl (fd, SIOCGIFINDEX, ifr) < 0)
        {
          unsigned int j;
          err = errno;

          for (j =  0; j < i; ++j)
            free (idx[j].if_name);
          free (idx);
	  idx = NULL;

          if (err == EINVAL)
            err = ENOSYS;
          else if (err == ENOMEM)
            err = ENOBUFS;
          goto out;
        }
      idx[i].if_index = ifr->ifr_ifindex;
    }

  idx[i].if_index = 0;
  idx[i].if_name = NULL;

 out:
  __close (fd);
  if (data != ifc.ifc_buf)
    __vm_deallocate (__mach_task_self (), (vm_address_t) ifc.ifc_buf,
		     ifc.ifc_len);
  __set_errno (err);
  return idx;
}
weak_alias (__if_nameindex, if_nameindex)
libc_hidden_weak (if_nameindex)

/* Store the name of the interface corresponding to index IFINDEX in
   IFNAME (which has space for at least IFNAMSIZ characters).  Return
   IFNAME, or NULL on error.  */
char *
__if_indextoname (unsigned int ifindex, char ifname[IF_NAMESIZE])
{
  struct ifreq ifr;
  int fd = __opensock ();

  if (fd < 0)
    return NULL;

  ifr.ifr_ifindex = ifindex;
  if (__ioctl (fd, SIOCGIFNAME, &ifr) < 0)
    {
      int saved_errno = errno;
      __close (fd);
      if (saved_errno == EINVAL || saved_errno == ENOTTY)
        __set_errno (ENOSYS);
      else if (saved_errno == ENODEV)
	__set_errno (ENXIO);
      return NULL;
    }
  __close (fd);
  return strncpy (ifname, ifr.ifr_name, IFNAMSIZ);
}
weak_alias (__if_indextoname, if_indextoname)
libc_hidden_weak (if_indextoname)

#if 0
void
__protocol_available (int *have_inet, int *have_inet6)
{
  *have_inet = _hurd_socket_server (PF_INET, 0) != MACH_PORT_NULL;
  *have_inet6 = _hurd_socket_server (PF_INET6, 0) != MACH_PORT_NULL;
}
#endif
