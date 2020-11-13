dnl Nesting safe macros for saving variables
dnl Usage: PAC_PUSH_FLAG(CFLAGS)
AC_DEFUN([PAC_PUSH_FLAG],[
	if test -z "${pac_save_$1_nesting}" ; then
	   pac_save_$1_nesting=0
	fi
	eval pac_save_$1_${pac_save_$1_nesting}='"$$1"'
	pac_save_$1_nesting=`expr ${pac_save_$1_nesting} + 1`
])

dnl Usage: PAC_POP_FLAG(CFLAGS)
AC_DEFUN([PAC_POP_FLAG],[
	pac_save_$1_nesting=`expr ${pac_save_$1_nesting} - 1`
	eval $1="\$pac_save_$1_${pac_save_$1_nesting}"
	eval pac_save_$1_${pac_save_$1_nesting}=""
])

dnl Usage: PAC_PUSH_ALL_FLAGS
AC_DEFUN([PAC_PUSH_ALL_FLAGS],[
	PAC_PUSH_FLAG(CFLAGS)
	PAC_PUSH_FLAG(CPPFLAGS)
	PAC_PUSH_FLAG(CXXFLAGS)
	PAC_PUSH_FLAG(FFLAGS)
	PAC_PUSH_FLAG(FCFLAGS)
	PAC_PUSH_FLAG(LDFLAGS)
	PAC_PUSH_FLAG(LIBS)
])

dnl Usage: PAC_POP_ALL_FLAGS
AC_DEFUN([PAC_POP_ALL_FLAGS],[
	PAC_POP_FLAG(CFLAGS)
	PAC_POP_FLAG(CPPFLAGS)
	PAC_POP_FLAG(CXXFLAGS)
	PAC_POP_FLAG(FFLAGS)
	PAC_POP_FLAG(FCFLAGS)
	PAC_POP_FLAG(LDFLAGS)
	PAC_POP_FLAG(LIBS)
])

dnl PAC_PREFIX_FLAG - Save flag with a prefix
dnl Usage: PAC_PREFIX_FLAG(PREFIX, FLAG)
AC_DEFUN([PAC_PREFIX_FLAG],[
	$1_$2=$$2
	export $1_$2
	AC_SUBST($1_$2)
])

dnl PAC_PREFIX_ALL_FLAGS - Save flags with a prefix
dnl Usage: PAC_PREFIX_ALL_FLAGS(PREFIX)
AC_DEFUN([PAC_PREFIX_ALL_FLAGS],[
	PAC_PREFIX_FLAG($1, CFLAGS)
	PAC_PREFIX_FLAG($1, CPPFLAGS)
	PAC_PREFIX_FLAG($1, CXXFLAGS)
	PAC_PREFIX_FLAG($1, FFLAGS)
	PAC_PREFIX_FLAG($1, FCFLAGS)
	PAC_PREFIX_FLAG($1, LDFLAGS)
	PAC_PREFIX_FLAG($1, LIBS)
	PAC_PREFIX_FLAG($1, EXTRA_LIBS)
])

dnl Usage: PAC_APPEND_FLAG([-02], [CFLAGS])
dnl appends the given argument to the specified shell variable unless the
dnl argument is already present in the variable
AC_DEFUN([PAC_APPEND_FLAG],[
	AC_REQUIRE([AC_PROG_FGREP])
	AS_IF(
		[echo "$$2" | $FGREP -e '$1' >/dev/null 2>&1],
		[echo "$2(='$$2') contains '$1', not appending" >&AS_MESSAGE_LOG_FD],
		[echo "$2(='$$2') does not contain '$1', appending" >&AS_MESSAGE_LOG_FD
		$2="$$2 $1"]
	)
])

dnl Usage: PAC_PREPEND_FLAG([-lpthread], [LIBS])
dnl Prepends the given argument to the specified shell variable unless the
dnl argument is already present in the variable.
dnl
dnl This is typically used for LIBS and similar variables because libraries
dnl should be added in reverse order.
AC_DEFUN([PAC_PREPEND_FLAG],[
        AC_REQUIRE([AC_PROG_FGREP])
        AS_IF(
                [echo "$$2" | $FGREP -e '$1' >/dev/null 2>&1],
                [echo "$2(='$$2') contains '$1', not prepending" >&AS_MESSAGE_LOG_FD],
                [echo "$2(='$$2') does not contain '$1', prepending" >&AS_MESSAGE_LOG_FD
                $2="$1 $$2"]
        )
])


dnl PAC_MKDIRS(path)
dnl Create any missing directories in the path
AC_DEFUN([PAC_MKDIRS],[
# Build any intermediate directories
for dir in $1 ; do
    PAC_PUSH_FLAG([IFS])
    IFS="/"
    tmp_curdir=""
    for tmp_subdir in $dir ; do
	tmp_curdir="${tmp_curdir}$tmp_subdir"
	if test ! -d "$tmp_curdir" ; then mkdir "$tmp_curdir" ; fi
        tmp_curdir="${tmp_curdir}/"
    done
    PAC_POP_FLAG([IFS])
done
])

# Find something to use for mkdir -p.  Eventually, this will have a
# script for backup. As of autoconf-2.63, AC_PROG_MKDIR_P was broken;
# it was checking to see if it recognized the "version" of mkdir and
# was deciding based on that. This should always be a feature test.
AC_DEFUN([PAC_PROG_MKDIR_P],[
AC_CACHE_CHECK([whether mkdir -p works],
pac_cv_mkdir_p,[
pac_cv_mkdir_p=no
rm -rf .tmp
if mkdir -p .tmp/.foo 1>/dev/null 2>&1 ; then
    if test -d .tmp/.foo ; then
        pac_cv_mkdir_p=yes
    fi
fi
rm -rf .tmp
])
if test "$pac_cv_mkdir_p" = "yes" ; then
   MKDIR_P="mkdir -p"
   export MKDIR_P
else
   AC_MSG_WARN([mkdir -p does not work; the install step may fail])
fi
AC_SUBST(MKDIR_P)
])

dnl Test for a clean VPATH directory.  Provide this command with the names
dnl of all of the generated files that might cause problems 
dnl (Makefiles won't cause problems because there's no VPATH usage for them)
dnl
dnl Synopsis
dnl PAC_VPATH_CHECK([file-names],[directory-names])
dnl  file-names should be files other than config.status and any header (e.g.,
dnl fooconf.h) file that should be removed.  It is optional
AC_DEFUN([PAC_VPATH_CHECK],[
# This is needed for Mac OSX 10.5
rm -rf conftest.dSYM
rm -f conftest*
date >conftest$$
# If creating a file in the current directory does not show up in the srcdir
# then we're doing a VPATH build (or something is very wrong)
if test ! -s $srcdir/conftest$$ ; then
    pac_dirtyfiles=""
    pac_dirtydirs=""
    pac_header=""
    ifdef([AC_LIST_HEADER],[pac_header=AC_LIST_HEADER])
    for file in config.status $pac_header $1 ; do
        if test -f $srcdir/$file ; then 
	    pac_dirtyfiles="$pac_dirtyfiles $file"
	fi
    done
    ifelse($2,,,[
 	for dir in $2 ; do 
            if test -d $srcdir/$dir ; then
                pac_dirtydirs="$pac_dirtydirs $dir"
	    fi
	done
    ])

    if test -n "$pac_dirtyfiles" -o -n "$pac_dirtydirs" ; then
	# Create a nice message about what to remove
	rmmsg=""
	if test -n "$pac_dirtyfiles" ; then
	    rmmsg="files $pac_dirtyfiles"
        fi
 	if test -n "$pac_dirtydirs" ; then
	    if test -n "$rmmsg" ; then
	        rmmsg="$rmmsg and directories $pac_dirtydirs"
            else
                rmmsg="directories $pac_dirtydirs"
            fi
        fi
        if test -f $srcdir/Makefile ; then
            AC_MSG_ERROR([You cannot do a VPATH build if the source directory has been
    configured.  Run "make distclean" in $srcdir first and make sure that the
    $rmmsg have been removed.])
        else
            AC_MSG_ERROR([You cannot do a VPATH build if the source directory has been
    configured.  Remove the $rmmsg in $srcdir.])
        fi
    fi
fi
# This is needed for Mac OSX 10.5
rm -rf conftest.dSYM
rm -f conftest*
])

dnl PAC_CONF_HEX_TO_DEC(value,out_var)
dnl
dnl Converts the given hexadecimal integer constant to an integer constant and
dnl stores the result in the shell variable given by 'out_var'.
dnl
dnl I think that printf like this will be sufficiently portable, but I don't
dnl have any guarantee of it.  If not, we can fall back to AS_VAR_ARITH
dnl and/or AC_COMPUTE_INT (the latter will probably be slow)
AC_DEFUN([PAC_CONV_HEX_TO_DEC],[AS_VAR_SET([$2],[`printf "%d" $1`])])

dnl PAC_GET_EXENAME(exe_name, out_exe_name)
dnl
dnl Prepends and appends program prefix and suffix as supplied by --program_prefix
dnl and --program-sufix
AC_DEFUN([PAC_GET_EXENAME],[
$2=$1
if test "$program_prefix" != "NONE" ; then
    $2="${program_prefix}$$2"
fi
if test "$program_suffix" != "NONE" ; then
    $2="$$2$program_suffix"
fi
])
