GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/unix/sysv/linux.

define([LIBC_LINUX_VERSION],[3.2.0])dnl
if test -n "$sysheaders"; then
  OLD_CPPFLAGS=$CPPFLAGS
  CPPFLAGS="$CPPFLAGS $SYSINCLUDES"
fi
define([libc_cv_linuxVER], [libc_cv_linux]patsubst(LIBC_LINUX_VERSION,[\.]))dnl
AC_CACHE_CHECK(installed Linux kernel header files, libc_cv_linuxVER, [dnl
AC_PREPROC_IFELSE([AC_LANG_PROGRAM([[#include <linux/version.h>
#if !defined LINUX_VERSION_CODE || LINUX_VERSION_CODE < ]dnl
patsubst(LIBC_LINUX_VERSION,[^\([^.]*\)\.\([^.]*\)\.\([^.]*\)$],dnl
[ (\1 *65536+ \2 *256+ \3) /* \1.\2.\3 */])[
# error kernel headers missing or too old
#endif]], [[]])],
	       [libc_cv_linuxVER='LIBC_LINUX_VERSION or later'],
	       [libc_cv_linuxVER='missing or too old!'])])
if test "$libc_cv_linuxVER" != 'LIBC_LINUX_VERSION or later'; then
  AC_MSG_ERROR([GNU libc requires kernel header files from
Linux LIBC_LINUX_VERSION or later to be installed before configuring.
The kernel header files are found usually in /usr/include/asm and
/usr/include/linux; make sure these directories use files from
Linux LIBC_LINUX_VERSION or later.  This check uses <linux/version.h>, so
make sure that file was built correctly when installing the kernel header
files.  To use kernel headers not from /usr/include/linux, use the
configure option --with-headers.])
fi

# If the user gave a minimal version number test whether the available
# kernel headers are young enough.  Additionally we have minimal
# kernel versions for some architectures.  If a previous configure fragment
# set arch_minimum_kernel already, let that override our defaults here.
# Note that we presume such a fragment has set libc_cv_gcc_unwind_find_fde
# if appropriate too.
test -n "$arch_minimum_kernel" || arch_minimum_kernel=3.2.0
if test -n "$minimum_kernel"; then
  changequote(,)
  user_version=$((`echo "$minimum_kernel.0.0.0" | sed 's/\([0-9]*\)\.\([0-9]*\)\.\([0-9]*\).*/\1 \* 65536 + \2 \* 256 + \3/'`))
  arch_version=$((`echo "$arch_minimum_kernel.0.0.0" | sed 's/\([0-9]*\)\.\([0-9]*\)\.\([0-9]*\).*/\1 \* 65536 + \2 \* 256 + \3/'`))
  changequote([,])
  if test $user_version -lt $arch_version; then
    AC_MSG_WARN([minimum kernel version reset to $arch_minimum_kernel])
    minimum_kernel=$arch_minimum_kernel
  fi
else
  minimum_kernel=$arch_minimum_kernel
fi

AC_MSG_CHECKING(for kernel header at least $minimum_kernel)
changequote(,)dnl
decnum=`echo "$minimum_kernel.0.0.0" | sed 's/\([0-9]*\)\.\([0-9]*\)\.\([0-9]*\).*/(\1 * 65536 + \2 * 256 + \3)/'`;
abinum=`echo "$minimum_kernel.0.0.0" | sed 's/\([0-9]*\)\.\([0-9]*\)\.\([0-9]*\).*/\1,\2,\3/'`;
changequote([,])dnl
AC_PREPROC_IFELSE([AC_LANG_PROGRAM([[#include <linux/version.h>
#if LINUX_VERSION_CODE < $decnum
# error kernel headers too old
#endif]], [[]])], [libc_minimum_kernel=ok], [libc_minimum_kernel='too old!'])
AC_MSG_RESULT($libc_minimum_kernel)
if test "$libc_minimum_kernel" = ok; then
  AC_DEFINE_UNQUOTED(__LINUX_KERNEL_VERSION, $decnum)
  AC_DEFINE_UNQUOTED(__ABI_TAG_VERSION, $abinum)
else
  AC_MSG_ERROR([*** The available kernel headers are older than the requested
*** compatible kernel version])
fi

if test -n "$sysheaders"; then
  CPPFLAGS=$OLD_CPPFLAGS
fi

if test "$prefix" = "/usr/local" -o "$prefix" = "/usr/local/" -o "$prefix" = "NONE"; then
  if test $enable_sanity = yes; then
    echo "\
*** On GNU/Linux systems the GNU C Library should not be installed into
*** /usr/local since this might make your system totally unusable.
*** We strongly advise to use a different prefix.  For details read the FAQ.
*** If you really mean to do this, run configure again using the extra
*** parameter \`--disable-sanity-checks'."
    exit 1
  else
    echo "\
*** WARNING: Do you really want to install the GNU C Library into /usr/local?
*** This might make your system totally unusable, for details read the FAQ."
  fi
fi


# One Linux we use ldconfig.
use_ldconfig=yes

if test $host = $build; then
  # If $prefix/include/{net,scsi} are symlinks, make install will
  # clobber what they're linked to (probably a kernel tree).
  # test -L ought to work on all Linux boxes.
  if test "x$prefix" != xNONE; then
    ac_prefix=$prefix
  else
    ac_prefix=$ac_default_prefix
  fi
  AC_MSG_CHECKING([for symlinks in ${ac_prefix}/include])
  ac_message=
  if test -L ${ac_prefix}/include/net; then
    ac_message="$ac_message
   ${ac_prefix}/include/net is a symlink"
  fi
  if test -L ${ac_prefix}/include/scsi; then
    ac_message="$ac_message
   ${ac_prefix}/include/scsi is a symlink"
  fi
  if test -n "$ac_message"; then
    AC_MSG_ERROR([$ac_message
\`make install' will destroy the target of the link(s).
Delete the links and re-run configure, or better still, move the entire
${ac_prefix}/include directory out of the way.])
  else
    AC_MSG_RESULT(ok)
  fi
fi

# We have inlined syscalls.
AC_DEFINE(HAVE_INLINED_SYSCALLS)
