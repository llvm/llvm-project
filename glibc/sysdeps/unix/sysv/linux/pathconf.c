/* Get file-specific information about a file.  Linux version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <mntent.h>
#include <stdio_ext.h>
#include <string.h>
#include <unistd.h>
#include <sys/sysmacros.h>

#include "pathconf.h"
#include "linux_fsinfo.h"
#include <not-cancel.h>

static long int posix_pathconf (const char *file, int name);

/* Define this first, so it can be inlined.  */
#define __pathconf static posix_pathconf
#include <sysdeps/posix/pathconf.c>


/* Get file-specific information about FILE.  */
long int
__pathconf (const char *file, int name)
{
  struct statfs fsbuf;

  switch (name)
    {
    case _PC_LINK_MAX:
      return __statfs_link_max (__statfs (file, &fsbuf), &fsbuf, file, -1);

    case _PC_FILESIZEBITS:
      return __statfs_filesize_max (__statfs (file, &fsbuf), &fsbuf);

    case _PC_2_SYMLINKS:
      return __statfs_symlinks (__statfs (file, &fsbuf), &fsbuf);

    case _PC_CHOWN_RESTRICTED:
      return __statfs_chown_restricted (__statfs (file, &fsbuf), &fsbuf);

    default:
      return posix_pathconf (file, name);
    }
}


static long int
distinguish_extX (const struct statfs *fsbuf, const char *file, int fd)
{
  char buf[64];
  char path[PATH_MAX];
  struct __stat64_t64 st;

  if ((file == NULL ? __fstat64_time64 (fd, &st)
		    : __stat64_time64 (file, &st)) != 0)
    /* Strange.  The statfd call worked, but stat fails.  Default to
       the more pessimistic value.  */
    return EXT2_LINK_MAX;

  __snprintf (buf, sizeof (buf), "/sys/dev/block/%u:%u",
	      __gnu_dev_major (st.st_dev), __gnu_dev_minor (st.st_dev));

  ssize_t n = __readlink (buf, path, sizeof (path));
  if (n != -1 && n < sizeof (path))
    {
      path[n] = '\0';
      char *base = strdupa (__basename (path));
      __snprintf (path, sizeof (path), "/sys/fs/ext4/%s", base);

      return __access (path, F_OK) == 0 ? EXT4_LINK_MAX : EXT2_LINK_MAX;
    }

  /* XXX Is there a better way to distinguish ext2/3 from ext4 than
     iterating over the mounted filesystems and compare the device
     numbers?  */
  FILE *mtab = __setmntent ("/proc/mounts", "r");
  if (mtab == NULL)
    mtab = __setmntent (_PATH_MOUNTED, "r");

  /* By default be conservative.  */
  long int result = EXT2_LINK_MAX;
  if (mtab != NULL)
    {
      struct mntent mntbuf;
      char tmpbuf[1024];

      /* No locking needed.  */
      (void) __fsetlocking (mtab, FSETLOCKING_BYCALLER);

      while (__getmntent_r (mtab, &mntbuf, tmpbuf, sizeof (tmpbuf)))
	{
	  if (strcmp (mntbuf.mnt_type, "ext2") != 0
	      && strcmp (mntbuf.mnt_type, "ext3") != 0
	      && strcmp (mntbuf.mnt_type, "ext4") != 0)
	    continue;

	  struct stat64 fsst;
	  if (__stat64 (mntbuf.mnt_dir, &fsst) >= 0
	      && st.st_dev == fsst.st_dev)
	    {
	      if (strcmp (mntbuf.mnt_type, "ext4") == 0)
		result = EXT4_LINK_MAX;
	      break;
	    }
	}

      /* Close the file.  */
      __endmntent (mtab);
    }

  return result;
}


/* Used like: return statfs_link_max (__statfs (name, &buf), &buf); */
long int
__statfs_link_max (int result, const struct statfs *fsbuf, const char *file,
		   int fd)
{
  if (result < 0)
    {
      if (errno == ENOSYS)
	/* Not possible, return the default value.  */
	return LINUX_LINK_MAX;

      /* Some error occured.  */
      return -1;
    }

  switch (fsbuf->f_type)
    {
    case EXT2_SUPER_MAGIC:
      /* Unfortunately the kernel does not return a different magic number
	 for ext4.  This would be necessary to easily detect etx4 since it
	 has a different LINK_MAX value.  Therefore we have to find it out
	 the hard way.  */
      return distinguish_extX (fsbuf, file, fd);

    case F2FS_SUPER_MAGIC:
      return F2FS_LINK_MAX;

    case MINIX_SUPER_MAGIC:
    case MINIX_SUPER_MAGIC2:
      return MINIX_LINK_MAX;

    case MINIX2_SUPER_MAGIC:
    case MINIX2_SUPER_MAGIC2:
      return MINIX2_LINK_MAX;

    case XENIX_SUPER_MAGIC:
      return XENIX_LINK_MAX;

    case SYSV4_SUPER_MAGIC:
    case SYSV2_SUPER_MAGIC:
      return SYSV_LINK_MAX;

    case COH_SUPER_MAGIC:
      return COH_LINK_MAX;

    case UFS_MAGIC:
    case UFS_CIGAM:
      return UFS_LINK_MAX;

    case REISERFS_SUPER_MAGIC:
      return REISERFS_LINK_MAX;

    case XFS_SUPER_MAGIC:
      return XFS_LINK_MAX;

    case LUSTRE_SUPER_MAGIC:
      return LUSTRE_LINK_MAX;

    default:
      return LINUX_LINK_MAX;
    }
}


/* Used like: return statfs_filesize_max (__statfs (name, &buf), &buf); */
long int
__statfs_filesize_max (int result, const struct statfs *fsbuf)
{
  if (result < 0)
    {
      if (errno == ENOSYS)
	/* Not possible, return the default value.  */
	return 32;

      /* Some error occured.  */
      return -1;
    }

  switch (fsbuf->f_type)
    {
    case F2FS_SUPER_MAGIC:
      return 256;

    case BTRFS_SUPER_MAGIC:
      return 255;

    case EXT2_SUPER_MAGIC:
    case UFS_MAGIC:
    case UFS_CIGAM:
    case REISERFS_SUPER_MAGIC:
    case XFS_SUPER_MAGIC:
    case SMB_SUPER_MAGIC:
    case NTFS_SUPER_MAGIC:
    case UDF_SUPER_MAGIC:
    case JFS_SUPER_MAGIC:
    case VXFS_SUPER_MAGIC:
    case CGROUP_SUPER_MAGIC:
    case LUSTRE_SUPER_MAGIC:
      return 64;

    case MSDOS_SUPER_MAGIC:
    case JFFS_SUPER_MAGIC:
    case JFFS2_SUPER_MAGIC:
    case NCP_SUPER_MAGIC:
    case ROMFS_SUPER_MAGIC:
      return 32;

    default:
      return 32;
    }
}


/* Used like: return statfs_link_max (__statfs (name, &buf), &buf); */
long int
__statfs_symlinks (int result, const struct statfs *fsbuf)
{
  if (result < 0)
    {
      if (errno == ENOSYS)
	/* Not possible, return the default value.  */
	return 1;

      /* Some error occured.  */
      return -1;
    }

  switch (fsbuf->f_type)
    {
    case ADFS_SUPER_MAGIC:
    case BFS_MAGIC:
    case CRAMFS_MAGIC:
    case DEVPTS_SUPER_MAGIC:
    case EFS_SUPER_MAGIC:
    case EFS_MAGIC:
    case MSDOS_SUPER_MAGIC:
    case NTFS_SUPER_MAGIC:
    case QNX4_SUPER_MAGIC:
    case ROMFS_SUPER_MAGIC:
      /* No symlink support.  */
      return 0;

    default:
      return 1;
    }
}


/* Used like: return __statfs_chown_restricted (__statfs (name, &buf), &buf);*/
long int
__statfs_chown_restricted (int result, const struct statfs *fsbuf)
{
  if (result < 0)
    {
      if (errno == ENOSYS)
	/* Not possible, return the default value.  */
	return 1;

      /* Some error occured.  */
      return -1;
    }

  return 1;
}
