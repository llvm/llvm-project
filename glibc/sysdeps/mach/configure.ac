GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.

AC_CHECK_TOOL(MIG, mig, MISSING)
if test "x$MIG" = xMISSING; then
  AC_MSG_ERROR([cannot find required build tool mig])
fi
LIBC_CONFIG_VAR([MIG], [$MIG])

if test -n "$sysheaders"; then
  OLD_CPPFLAGS=$CPPFLAGS
  CPPFLAGS="$CPPFLAGS $SYSINCLUDES"
fi

### Sanity checks for Mach header installation
AC_CHECK_HEADER(mach/mach_types.h,,
                [AC_MSG_ERROR([cannot find Mach headers])], -)
AC_CHECK_HEADER(mach/mach_types.defs,, [dnl
AC_MSG_ERROR([cannot find Mach .defs files])], -)

dnl
dnl mach_TYPE_CHECK(foo_t, bar_t)
dnl
dnl Check if foo_t is defined by <mach/mach_types.h>.
dnl If not, compile with -Dfoo_t=bar_t.
dnl
AC_DEFUN([mach_TYPE_CHECK], [dnl
AC_CACHE_CHECK(for $1 in mach/mach_types.h, libc_cv_mach_$1,
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <mach/mach_types.h>]], [[extern $1 foo;]])],
[libc_cv_mach_$1=$1], [libc_cv_mach_$1=$2]))
if test [$]libc_cv_mach_$1 != $1; then
  DEFINES="$DEFINES -D$1=$2"
fi])

dnl
dnl OSF Mach has renamed these typedefs for some reason.
dnl
mach_TYPE_CHECK(task_t, task_port_t)
mach_TYPE_CHECK(thread_t, thread_port_t)

dnl
dnl The creation_time field is a GNU Mach addition the other variants lack.
dnl
AC_CACHE_CHECK(for creation_time in task_basic_info,
	       libc_cv_mach_task_creation_time, [dnl
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <mach/task_info.h>]], [[
extern struct task_basic_info *i;
long s = i->creation_time.seconds;
]])], [libc_cv_mach_task_creation_time=yes], [libc_cv_mach_task_creation_time=no])])
if test $libc_cv_mach_task_creation_time = no; then
  AC_MSG_ERROR([you need Mach headers supporting task_info.creation_time])
fi

dnl
dnl The Darwin variant no longer has <mach/mach.defs>
dnl but instead has several constituent .defs files.
dnl In this scenario we will presume there is a <mach/mach_interface.h>
dnl that contains an #include for each constituent header file,
dnl but we don't do a check for that here because in a bare
dnl environment the compile against those headers will fail.
dnl
mach_interface_list=
for ifc in mach mach4 gnumach \
	   clock clock_priv host_priv host_security ledger lock_set \
	   processor processor_set task task_notify thread_act vm_map \
	   memory_object memory_object_default default_pager \
	   i386/mach_i386 \
	   ; do
  AC_CHECK_HEADER(mach/${ifc}.defs, [dnl
  mach_interface_list="$mach_interface_list $ifc"],, -)
done
if test "x$mach_interface_list" = x; then
  AC_MSG_ERROR([what manner of Mach is this?])
fi

AC_CACHE_CHECK(for host_page_size in mach_host.defs,
	       libc_cv_mach_host_page_size, [dnl
AC_EGREP_HEADER(host_page_size, mach/mach_host.defs,
		libc_cv_mach_host_page_size=yes,
		libc_cv_mach_host_page_size=no)])
if test $libc_cv_mach_host_page_size = yes; then
  AC_DEFINE([HAVE_HOST_PAGE_SIZE])
fi

AC_CHECK_HEADER(mach/machine/ndr_def.h, [dnl
  DEFINES="$DEFINES -DNDR_DEF_HEADER='<mach/machine/ndr_def.h>'"], [dnl
AC_CHECK_HEADER(machine/ndr_def.h, [dnl
  DEFINES="$DEFINES -DNDR_DEF_HEADER='<machine/ndr_def.h>'"],, -)], -)

AC_CACHE_CHECK(for i386_io_perm_modify in mach_i386.defs,
	       libc_cv_mach_i386_ioports, [dnl
AC_EGREP_HEADER(i386_io_perm_modify, mach/i386/mach_i386.defs,
		libc_cv_mach_i386_ioports=yes,
		libc_cv_mach_i386_ioports=no)])
if test $libc_cv_mach_i386_ioports = yes; then
  AC_DEFINE([HAVE_I386_IO_PERM_MODIFY])
fi

AC_CACHE_CHECK(for i386_set_gdt in mach_i386.defs,
	       libc_cv_mach_i386_gdt, [dnl
AC_EGREP_HEADER(i386_set_gdt, mach/i386/mach_i386.defs,
		libc_cv_mach_i386_gdt=yes,
		libc_cv_mach_i386_gdt=no)])
if test $libc_cv_mach_i386_gdt = yes; then
  AC_DEFINE([HAVE_I386_SET_GDT])
fi

dnl Swiped from hurd/aclocal.m4
AC_DEFUN([hurd_MIG_RETCODE], [dnl
# See if mig groks `retcode'.
AC_CACHE_CHECK(whether $MIG supports the retcode keyword, hurd_cv_mig_retcode,
[cat > conftest.defs <<\EOF
#include <mach/std_types.defs>
#include <mach/mach_types.defs>
subsystem foobar 1000;
type reply_port_t = polymorphic | MACH_MSG_TYPE_PORT_SEND_ONCE
	ctype: mach_port_t;
simpleroutine foobar_reply (
	reply_port: reply_port_t;
	err: kern_return_t, RetCode);
EOF
if AC_TRY_COMMAND([CC="${CC}" ${MIG-false} -n conftest.defs 1>&AS_MESSAGE_LOG_FD]); then
  hurd_cv_mig_retcode=yes
else
  hurd_cv_mig_retcode=no
fi
rm -f conftest*])
if test $hurd_cv_mig_retcode = yes; then
  AC_DEFINE(HAVE_MIG_RETCODE)
fi])

hurd_MIG_RETCODE

if test -n "$sysheaders"; then
  CPPFLAGS=$OLD_CPPFLAGS
fi
