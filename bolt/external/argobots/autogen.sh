#! /bin/sh
#
# See COPYRIGHT in top-level directory.
#

########################################################################
## Utility functions
########################################################################

recreate_tmp() {
    rm -rf .tmp
    mkdir .tmp 2>&1 >/dev/null
}

warn() {
    echo "===> WARNING: $@"
}

error() {
    echo "===> ERROR:   $@"
}

echo_n() {
    # "echo -n" isn't portable, must portably implement with printf
    printf "%s" "$*"
}


echo "================================================================="
echo "Checking autotools installations"
echo "================================================================="

########################################################################
## Verify autoconf version
########################################################################

echo_n "Checking for autoconf version... "
recreate_tmp
ver=2.67
cat > .tmp/configure.ac<<EOF
AC_INIT
AC_PREREQ($ver)
AC_OUTPUT
EOF
if (cd .tmp && $autoreconf $autoreconf_args >/dev/null 2>&1 ) ; then
    echo ">= $ver"
else
    echo "bad autoconf installation"
    cat <<EOF
You either do not have autoconf in your path or it is too old (version
$ver or higher required). You may be able to use

     autoconf --version

Unfortunately, there is no standard format for the version output and
it changes between autotools versions.  In addition, some versions of
autoconf choose among many versions and provide incorrect output).
EOF
    exit 1
fi


########################################################################
## Verify automake version
########################################################################

echo_n "Checking for automake version... "
recreate_tmp
ver=1.12.3
cat > .tmp/configure.ac<<EOF
AC_INIT(testver,1.0)
AC_CONFIG_AUX_DIR([m4])
AC_CONFIG_MACRO_DIR([m4])
m4_ifdef([AM_INIT_AUTOMAKE],,[m4_fatal([AM_INIT_AUTOMAKE not defined])])
AM_INIT_AUTOMAKE([$ver foreign])
AC_MSG_RESULT([A message])
AC_OUTPUT([Makefile])
EOF
cat <<EOF >.tmp/Makefile.am
ACLOCAL_AMFLAGS = -I m4
EOF
if [ ! -d .tmp/m4 ] ; then mkdir .tmp/m4 >/dev/null 2>&1 ; fi
if (cd .tmp && $autoreconf $autoreconf_args >/dev/null 2>&1 ) ; then
    echo ">= $ver"
else
    echo "bad automake installation"
    cat <<EOF
You either do not have automake in your path or it is too old (version
$ver or higher required). You may be able to use

     automake --version

Unfortunately, there is no standard format for the version output and
it changes between autotools versions.  In addition, some versions of
autoconf choose among many versions and provide incorrect output).
EOF
    exit 1
fi

########################################################################
## Verify libtool version
########################################################################

echo_n "Checking for libtool version... "
recreate_tmp
ver=2.4
cat <<EOF >.tmp/configure.ac
AC_INIT(testver,1.0)
AC_CONFIG_AUX_DIR([m4])
AC_CONFIG_MACRO_DIR([m4])
m4_ifdef([LT_PREREQ],,[m4_fatal([LT_PREREQ not defined])])
LT_PREREQ($ver)
LT_INIT()
AC_MSG_RESULT([A message])
EOF
cat <<EOF >.tmp/Makefile.am
ACLOCAL_AMFLAGS = -I m4
EOF
if [ ! -d .tmp/m4 ] ; then mkdir .tmp/m4 >/dev/null 2>&1 ; fi
if (cd .tmp && $autoreconf $autoreconf_args >/dev/null 2>&1 ) ; then
    echo ">= $ver"
else
    echo "bad libtool installation"
    cat <<EOF
You either do not have libtool in your path or it is too old
(version $ver or higher required). You may be able to use

     libtool --version

Unfortunately, there is no standard format for the version output and
it changes between autotools versions.  In addition, some versions of
autoconf choose among many versions and provide incorrect output).
EOF
    exit 1
fi


echo
echo "================================================================="
echo "Generating required temporary files"
echo "================================================================="

########################################################################
## Building maint/Version
########################################################################

# build a substitute maint/Version script now that we store the single copy of
# this information in an m4 file for autoconf's benefit
echo_n "Generating a helper maint/Version... "
if autom4te -l M4sugar maint/Version.base.m4 > maint/Version ; then
    echo "done"
else
    echo "error"
    error "unable to correctly generate maint/Version shell helper"
fi

########################################################################
## Building the README
########################################################################

echo_n "Updating the README... "
. ./maint/Version
if [ -f README.md ] ; then
    sed -e "s/%VERSION%/${ABT_VERSION}/g" README.md > README
    echo "done"
else
    echo "error"
    error "README.md file not present, unable to update README version number (perhaps we are running in a release tarball source tree?)"
fi


echo
echo "================================================================="
echo "Generating configure and friends"
echo "================================================================="

autoreconf -vif
