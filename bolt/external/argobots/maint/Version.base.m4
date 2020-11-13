dnl
dnl this m4 file expects to be processed with "autom4te -l M4sugar"
dnl
m4_init

dnl get the real version values
m4_include([maint/version.m4])dnl

dnl The m4sugar langauage switches the default diversion to "KILL", and causes
dnl all normal output to be thrown away.  We switch to the default (0) diversion
dnl to restore output.
m4_divert_push([])dnl
dnl
dnl now dump out a shell script header for those looking to change the version string
# This shell script is no longer the canonical version number script, but rather
# a byproduct of running ./maint/updatefiles using maint/Version.base.m4 as
# input.  The real version info is contained in maint/version.m4 instead.  It is
# intentionally not managed by the makefile system and may not be up to date at
# all times w.r.t. the version.m4 file.

dnl now provide shell versions so that simple scripts can still use
dnl $ABT_VERSION as before
ABT_VERSION=ABT_VERSION_m4
export ABT_VERSION

dnl balance our pushed diversion
m4_divert_pop([])dnl
