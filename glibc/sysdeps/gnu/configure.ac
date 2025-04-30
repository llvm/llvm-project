GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.

# Local configure fragment for sysdeps/gnu.

# The Filesystem Hierarchy Standard prescribes where to place "essential"
# files.  I.e., when the installation prefix is "/usr" we have to place
# shared library objects and the configuration files on the root partition
# in /lib and /etc.
LIBC_SLIBDIR_RTLDDIR([lib], [lib])
case "$prefix" in
/usr | /usr/)
  # Allow the user to override the path with --sysconfdir.
  if test "$sysconfdir" = '${prefix}/etc'; then
    libc_cv_sysconfdir=/etc
  else
    libc_cv_sysconfdir=$sysconfdir
   fi
  # Allow the user to override the path with --localstatedir.
  if test "$localstatedir" = '${prefix}/var'; then
    libc_cv_localstatedir=/var
  else
    libc_cv_localstatedir=$localstatedir
   fi
  libc_cv_rootsbindir=/sbin
  ;;
esac
