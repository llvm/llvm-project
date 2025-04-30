# Copyright (C) 1991-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

#
#	Master Makefile for the GNU C library
#
ifneq (,)
This makefile requires GNU Make.
endif

include Makeconfig


# This is the default target; it makes everything except the tests.
.PHONY: all help minihelp
all: minihelp lib others

help:
	@sed '0,/^help-starts-here$$/d' Makefile.help

minihelp:
	@echo
	@echo type \"make help\" for help with common glibc makefile targets
	@echo


ifneq ($(AUTOCONF),no)

define autoconf-it
@-rm -f $@.new
$(AUTOCONF) $(ACFLAGS) $< > $@.new
chmod a-w$(patsubst %,$(comma)a+x,$(filter .,$(@D))) $@.new
mv -f $@.new $@
endef

configure: configure.ac aclocal.m4; $(autoconf-it)
%/configure: %/configure.ac aclocal.m4; $(autoconf-it)
%/preconfigure: %/preconfigure.ac aclocal.m4; $(autoconf-it)

endif # $(AUTOCONF) = no


# We don't want to run anything here in parallel.
.NOTPARALLEL:

# These are the targets that are made by making them in each subdirectory.
+subdir_targets	:= subdir_lib objects objs others subdir_mostlyclean	\
		   subdir_clean subdir_distclean subdir_realclean	\
		   tests xtests						\
		   subdir_update-abi subdir_check-abi			\
		   subdir_update-all-abi				\
		   subdir_echo-headers					\
		   subdir_install					\
		   subdir_objs subdir_stubs subdir_testclean		\
		   $(addprefix install-, no-libc.a bin lib data headers others)

headers := limits.h values.h features.h features-time64.h gnu-versions.h \
	   bits/xopen_lim.h gnu/libc-version.h stdc-predef.h \
	   bits/libc-header-start.h

echo-headers: subdir_echo-headers

# The headers are in the include directory.
subdir-dirs = include
vpath %.h $(subdir-dirs)

# What to install.
install-others = $(inst_includedir)/gnu/stubs.h
install-bin-script =

ifeq (yes,$(build-shared))
headers += gnu/lib-names.h
endif

include Makerules

ifeq ($(build-programs),yes)
others: $(addprefix $(objpfx),$(install-bin-script))
endif

# Install from subdirectories too.
install: subdir_install

# Explicit dependency so that `make install-headers' works
install-headers: install-headers-nosubdir

# Make sure that the dynamic linker is installed before libc.
$(inst_slibdir)/libc-$(version).so: elf/ldso_install

.PHONY: elf/ldso_install
elf/ldso_install:
	$(MAKE) -C $(@D) $(@F)

# Create links for shared libraries using the `ldconfig' program if possible.
# Ignore the error if we cannot update /etc/ld.so.cache.
ifeq (no,$(cross-compiling))
ifeq (yes,$(build-shared))
install:
	-test ! -x $(elf-objpfx)ldconfig || LC_ALL=C \
	  $(elf-objpfx)ldconfig $(addprefix -r ,$(install_root)) \
				$(slibdir) $(libdir)
ifneq (no,$(PERL))
ifeq (/usr,$(prefix))
ifeq (,$(install_root))
	LD_SO=$(ld.so-version) CC="$(CC)" $(PERL) scripts/test-installation.pl $(common-objpfx)
endif
endif
endif
endif
endif

# Build subdirectory lib objects.
lib-noranlib: subdir_lib

ifeq (yes,$(build-shared))
# Build the shared object from the PIC object library.
lib: $(common-objpfx)libc.so $(common-objpfx)linkobj/libc.so
endif # $(build-shared)

# Used to build testrun.sh.
define testrun-script
#!/bin/bash
builddir=`dirname "$$0"`
GCONV_PATH="$${builddir}/iconvdata"

usage () {
cat << EOF
Usage: $$0 [OPTIONS] <program> [ARGUMENTS...]

  --tool=TOOL  Run with the specified TOOL. It can be strace, valgrind or
               container. The container will run within support/test-container.
EOF

  exit 1
}

toolname=default
while test $$# -gt 0 ; do
  case "$$1" in
    --tool=*)
      toolname="$${1:7}"
      shift
      ;;
    --*)
      usage
      ;;
    *)
      break
      ;;
  esac
done

if test $$# -eq 0 ; then
  usage
fi

case "$$toolname" in
  default)
    exec $(subst $(common-objdir),"$${builddir}", $(test-program-prefix)) \
      $${1+"$$@"}
    ;;
  strace)
    exec strace $(patsubst %, -E%, $(run-program-env)) \
      $(test-via-rtld-prefix) $${1+"$$@"}
    ;;
  valgrind)
    exec env $(run-program-env) valgrind $(test-via-rtld-prefix) $${1+"$$@"}
    ;;
  container)
    exec env $(run-program-env) $(test-via-rtld-prefix) \
      $(common-objdir)/support/test-container \
      env $(run-program-env) $(test-via-rtld-prefix) $${1+"$$@"}
    ;;
  *)
    usage
    ;;
esac
endef

# This is a handy script for running any dynamically linked program against
# the current libc build for testing.
$(common-objpfx)testrun.sh: $(common-objpfx)config.make \
			    $(..)Makeconfig $(..)Makefile
	$(file >$@T,$(testrun-script))
	chmod a+x $@T
	mv -f $@T $@
postclean-generated += testrun.sh

define debugglibc
#!/bin/bash

SOURCE_DIR="$(CURDIR)"
BUILD_DIR="$(common-objpfx)"
CMD_FILE="$(common-objpfx)debugglibc.gdb"
CONTAINER=false
DIRECT=true
STATIC=false
SYMBOLSFILE=true
unset TESTCASE
unset BREAKPOINTS
unset ENVVARS

usage()
{
cat << EOF
Usage: $$0 [OPTIONS] <program>

   Or: $$0 [OPTIONS] -- <program> [<args>]...

  where <program> is the path to the program being tested,
  and <args> are the arguments to be passed to it.

Options:

  -h, --help
	Prints this message and leaves.

  The following options require one argument:

  -b, --breakpoint
	Breakpoints to set at the beginning of the execution
	(each breakpoint demands its own -b option, e.g. -b foo -b bar)
  -e, --environment-variable
	Environment variables to be set with 'exec-wrapper env' in GDB
	(each environment variable demands its own -e option, e.g.
	-e FOO=foo -e BAR=bar)

  The following options do not take arguments:

  -c, --in-container
	Run the test case inside a container and automatically attach
	GDB to it.
  -i, --no-direct
	Selects whether to pass the --direct flag to the program.
	--direct is useful when debugging glibc test cases. It inhibits the
	tests from forking and executing in a subprocess.
	Default behaviour is to pass the --direct flag, except when the
	program is run with user specified arguments using the "--" separator.
  -s, --no-symbols-file
	Do not tell GDB to load debug symbols from the program.
EOF
}

# Parse input options
while [[ $$# > 0 ]]
do
  key="$$1"
  case $$key in
    -h|--help)
      usage
      exit 0
      ;;
    -b|--breakpoint)
      BREAKPOINTS="break $$2\n$$BREAKPOINTS"
      shift
      ;;
    -e|--environment-variable)
      ENVVARS="$$2 $$ENVVARS"
      shift
      ;;
    -c|--in-container)
      CONTAINER=true
      ;;
    -i|--no-direct)
      DIRECT=false
      ;;
    -s|--no-symbols-file)
      SYMBOLSFILE=false
      ;;
    --)
      shift
      TESTCASE=$$1
      COMMANDLINE="$$@"
      # Don't add --direct when user specifies program arguments
      DIRECT=false
      break
      ;;
    *)
      TESTCASE=$$1
      COMMANDLINE=$$TESTCASE
      ;;
  esac
  shift
done

# Check for required argument and if the testcase exists
if [ ! -v TESTCASE ] || [ ! -f $${TESTCASE} ]
then
  usage
  exit 1
fi

# Container tests needing locale data should install them in-container.
# Other tests/binaries need to use locale data from the build tree.
if [ "$$CONTAINER" = false ]
then
  ENVVARS="GCONV_PATH=$${BUILD_DIR}/iconvdata $$ENVVARS"
  ENVVARS="LOCPATH=$${BUILD_DIR}/localedata $$ENVVARS"
  ENVVARS="LC_ALL=C $$ENVVARS"
fi

# Expand environment setup command
if [ -v ENVVARS ]
then
  ENVVARSCMD="set exec-wrapper env $$ENVVARS"
fi

# Expand direct argument
if [ "$$DIRECT" == true ]
then
  DIRECT="--direct"
else
  DIRECT=""
fi

# Check if the test case is static
if file $${TESTCASE} | grep "statically linked" >/dev/null
then
  STATIC=true
else
  STATIC=false
fi

# Expand symbols loading command
if [ "$$SYMBOLSFILE" == true ]
then
  SYMBOLSFILE="add-symbol-file $${TESTCASE}"
else
  SYMBOLSFILE=""
fi

# GDB commands template
template ()
{
cat <<EOF
set environment C -E -x c-header
set auto-load safe-path $${BUILD_DIR}/nptl_db:\$$debugdir:\$$datadir/auto-load
set libthread-db-search-path $${BUILD_DIR}/nptl_db
__ENVVARS__
__SYMBOLSFILE__
break _dl_start_user
run --library-path $(rpath-link):$${BUILD_DIR}/nptl_db \
__COMMANDLINE__ __DIRECT__
__BREAKPOINTS__
EOF
}

# Generate the commands file for gdb initialization
template | sed \
  -e "s|__ENVVARS__|$$ENVVARSCMD|" \
  -e "s|__SYMBOLSFILE__|$$SYMBOLSFILE|" \
  -e "s|__COMMANDLINE__|$$COMMANDLINE|" \
  -e "s|__DIRECT__|$$DIRECT|" \
  -e "s|__BREAKPOINTS__|$$BREAKPOINTS|" \
  > $$CMD_FILE

echo
echo "Debugging glibc..."
echo "Build directory  : $$BUILD_DIR"
echo "Source directory : $$SOURCE_DIR"
echo "GLIBC Testcase   : $$TESTCASE"
echo "GDB Commands     : $$CMD_FILE"
echo "Env vars         : $$ENVVARS"
echo

if [ "$$CONTAINER" == true ]
then
# Use testrun.sh to start the test case with WAIT_FOR_DEBUGGER=1, then
# automatically attach GDB to it.
WAIT_FOR_DEBUGGER=1 $(common-objpfx)testrun.sh --tool=container $${TESTCASE} &
gdb -x $${TESTCASE}.gdb
elif [ "$$STATIC" == true ]
then
gdb $${TESTCASE}
else
# Start the test case debugging in two steps:
#   1. the following command invokes gdb to run the loader;
#   2. the commands file tells the loader to run the test case.
gdb -q \
  -x $${CMD_FILE} \
  -d $${SOURCE_DIR} \
  $${BUILD_DIR}/elf/ld.so
fi
endef

# This is another handy script for debugging dynamically linked program
# against the current libc build for testing.
$(common-objpfx)debugglibc.sh: $(common-objpfx)config.make \
			    $(..)Makeconfig $(..)Makefile
	$(file >$@T,$(debugglibc))
	chmod a+x $@T
	mv -f $@T $@
postclean-generated += debugglibc.sh debugglibc.gdb

others: $(common-objpfx)testrun.sh $(common-objpfx)debugglibc.sh

# Makerules creates a file `stubs' in each subdirectory, which
# contains `#define __stub_FUNCTION' for each function defined in that
# directory which is a stub.
# Here we paste all of these together into <gnu/stubs.h>.

subdir-stubs := $(foreach dir,$(subdirs),$(common-objpfx)$(dir)/stubs)

ifndef abi-variants
installed-stubs = $(inst_includedir)/gnu/stubs.h
else
installed-stubs = $(inst_includedir)/gnu/stubs-$(default-abi).h

$(inst_includedir)/gnu/stubs.h: $(+force)
	$(make-target-directory)
	{ \
	 echo '/* This file is automatically generated.';\
	 echo "   This file selects the right generated file of \`__stub_FUNCTION' macros";\
	 echo '   based on the architecture being compiled for.  */'; \
	 echo ''; \
	 $(foreach h,$(abi-includes), echo '#include <$(h)>';) \
	 echo ''; \
	 $(foreach v,$(abi-variants),\
	 $(if $(abi-$(v)-condition),\
	 echo '#if $(abi-$(v)-condition)'; \
	 echo '# include <gnu/stubs-$(v).h>'); \
	 $(if $(abi-$(v)-condition),echo '#endif';) \
	 rm -f $(@:.d=.h).new$(v); \
	 ) \
	} > $(@:.d=.h).new
	mv -f $(@:.d=.h).new $(@:.d=.h)

install-others-nosubdir: $(installed-stubs)
endif


# Since stubs.h is never needed when building the library, we simplify the
# hairy installation process by producing it in place only as the last part
# of the top-level `make install'.  It depends on subdir_install, which
# iterates over all the subdirs; subdir_install in each subdir depends on
# the subdir's stubs file.  Having more direct dependencies would result in
# extra iterations over the list for subdirs and many recursive makes.
$(installed-stubs): include/stubs-prologue.h subdir_install
	$(make-target-directory)
	@rm -f $(objpfx)stubs.h
	(sed '/^@/d' $<; LC_ALL=C sort $(subdir-stubs)) > $(objpfx)stubs.h
	if test -r $@ && cmp -s $(objpfx)stubs.h $@; \
	then echo 'stubs.h unchanged'; \
	else $(INSTALL_DATA) $(objpfx)stubs.h $@; fi
	rm -f $(objpfx)stubs.h

# This makes the Info or DVI file of the documentation from the Texinfo source.
.PHONY: info dvi pdf html
info dvi pdf html:
	$(MAKE) $(PARALLELMFLAGS) -C manual $@

# This makes all the subdirectory targets.

# For each target, make it depend on DIR/target for each subdirectory DIR.
$(+subdir_targets): %: $(addsuffix /%,$(subdirs))

# Compute a list of all those targets.
all-subdirs-targets := $(foreach dir,$(subdirs),\
				 $(addprefix $(dir)/,$(+subdir_targets)))

# The action for each of those is to cd into the directory and make the
# target there.
$(all-subdirs-targets):
	$(MAKE) $(PARALLELMFLAGS) $(subdir-target-args) $(@F)

define subdir-target-args
subdir=$(@D)$(if $($(@D)-srcdir),\
-C $($(@D)-srcdir) ..=`pwd`/,\
-C $(@D) ..=../)
endef

.PHONY: $(+subdir_targets) $(all-subdirs-targets)

# Targets to clean things up to various degrees.

.PHONY: clean realclean distclean distclean-1 parent-clean parent-mostlyclean \
	tests-clean

# Subroutines of all cleaning targets.
parent-mostlyclean: common-mostlyclean # common-mostlyclean is in Makerules.
	-rm -f $(foreach o,$(object-suffixes-for-libc),\
		   $(common-objpfx)$(patsubst %,$(libtype$o),c)) \
	       $(addprefix $(objpfx),$(install-lib))
parent-clean: parent-mostlyclean common-clean

postclean = $(addprefix $(common-objpfx),$(postclean-generated)) \
	    $(addprefix $(objpfx),sysd-dirs sysd-rules) \
	    $(addprefix $(objpfx),sysd-sorted soversions.mk soversions.i)

clean: parent-clean
# This is done this way rather than having `subdir_clean' be a
# dependency of this target so that libc.a will be removed before the
# subdirectories are dealt with and so they won't try to remove object
# files from it when it's going to be removed anyway.
	@$(MAKE) subdir_clean no_deps=t
	-rm -f $(postclean)
mostlyclean: parent-mostlyclean
	@$(MAKE) subdir_mostlyclean no_deps=t
	-rm -f $(postclean)

tests-clean:
	@$(MAKE) subdir_testclean no_deps=t

ifneq (,$(CXX))
vpath c++-types.data $(+sysdep_dirs)

tests-special += $(objpfx)c++-types-check.out
$(objpfx)c++-types-check.out: c++-types.data scripts/check-c++-types.sh
	scripts/check-c++-types.sh $< $(CXX) $(filter-out -std=gnu11 $(+gccwarn-c),$(CFLAGS)) $(CPPFLAGS) > $@; \
	$(evaluate-test)
endif

tests-special += $(objpfx)check-local-headers.out
$(objpfx)check-local-headers.out: scripts/check-local-headers.sh
	AWK='$(AWK)' scripts/check-local-headers.sh \
	  "$(includedir)" "$(objpfx)" < /dev/null > $@; \
	$(evaluate-test)

ifneq "$(headers)" ""
# Special test of all the installed headers in this directory.
tests-special += $(objpfx)check-installed-headers-c.out
libof-check-installed-headers-c := testsuite
$(objpfx)check-installed-headers-c.out: \
    scripts/check-installed-headers.sh $(headers)
	$(SHELL) $(..)scripts/check-installed-headers.sh c \
	  "$(CC) $(filter-out -std=%,$(CFLAGS)) -D_ISOMAC $(+includes)" \
	  $(headers) > $@; \
	$(evaluate-test)

ifneq "$(CXX)" ""
tests-special += $(objpfx)check-installed-headers-cxx.out
libof-check-installed-headers-cxx := testsuite
$(objpfx)check-installed-headers-cxx.out: \
    scripts/check-installed-headers.sh $(headers)
	$(SHELL) $(..)scripts/check-installed-headers.sh c++ \
	  "$(CXX) $(filter-out -std=%,$(CXXFLAGS)) -D_ISOMAC $(+includes)" \
	  $(headers) > $@; \
	$(evaluate-test)
endif # $(CXX)

tests-special += $(objpfx)check-wrapper-headers.out
$(objpfx)check-wrapper-headers.out: scripts/check-wrapper-headers.py $(headers)
	$(PYTHON) $< --root=. --subdir=. $(headers) \
	  --generated $(common-generated) > $@; $(evaluate-test)
endif # $(headers)

define summarize-tests
@egrep -v '^(PASS|XFAIL):' $(objpfx)$1 || true
@echo "Summary of test results$2:"
@sed 's/:.*//' < $(objpfx)$1 | sort | uniq -c
@! egrep -q -v '^(X?PASS|XFAIL|UNSUPPORTED):' $(objpfx)$1
endef

# The intention here is to do ONE install of our build into the
# testroot.pristine/ directory, then rsync (internal to
# support/test-container) that to testroot.root/ at the start of each
# test.  That way we can promise each test a "clean" install, without
# having to do the install for each test.
#
# In addition, we have to copy some files (which we build) into this
# root in addition to what glibc installs.  For example, many tests
# require additional programs including /bin/sh, /bin/true, and
# /bin/echo, all of which we build below to limit library dependencies
# to just those things in glibc and language support libraries which
# we also copy into the into the rootfs.  To determine what language
# support libraries we need we build a "test" program in either C or
# (if available) C++ just so we can copy in any shared objects
# (which we do not build) that GCC-compiled programs depend on.


ifeq (,$(CXX))
LINKS_DSO_PROGRAM = links-dso-program-c
else
LINKS_DSO_PROGRAM = links-dso-program
endif

$(tests-container) $(addsuffix /tests,$(subdirs)) : \
		$(objpfx)testroot.pristine/install.stamp
$(objpfx)testroot.pristine/install.stamp :
	test -d $(objpfx)testroot.pristine || \
	  mkdir $(objpfx)testroot.pristine
	# We need a working /bin/sh for some of the tests.
	test -d $(objpfx)testroot.pristine/bin || \
	  mkdir $(objpfx)testroot.pristine/bin
	# We need the compiled locale dir for localedef tests.
	test -d $(objpfx)testroot.pristine/$(complocaledir) || \
	  mkdir -p $(objpfx)testroot.pristine/$(complocaledir)
	cp $(objpfx)support/shell-container $(objpfx)testroot.pristine/bin/sh
	cp $(objpfx)support/echo-container $(objpfx)testroot.pristine/bin/echo
	cp $(objpfx)support/true-container $(objpfx)testroot.pristine/bin/true
ifeq ($(run-built-tests),yes)
	# Copy these DSOs first so we can overwrite them with our own.
	for dso in `$(test-wrapper-env) LD_TRACE_LOADED_OBJECTS=1  \
		$(rtld-prefix) \
		$(objpfx)testroot.pristine/bin/sh \
	        | sed -n '/\//{s@.*=> /@/@;s/^[^/]*//;s/ .*//p;}'` ;\
	  do \
	    test -d `dirname $(objpfx)testroot.pristine$$dso` || \
	      mkdir -p `dirname $(objpfx)testroot.pristine$$dso` ;\
	    $(test-wrapper) cp $$dso $(objpfx)testroot.pristine$$dso ;\
	  done
	for dso in `$(test-wrapper-env) LD_TRACE_LOADED_OBJECTS=1  \
		$(rtld-prefix) \
		$(objpfx)support/$(LINKS_DSO_PROGRAM) \
	        | sed -n '/\//{s@.*=> /@/@;s/^[^/]*//;s/ .*//p;}'` ;\
	  do \
	    test -d `dirname $(objpfx)testroot.pristine$$dso` || \
	      mkdir -p `dirname $(objpfx)testroot.pristine$$dso` ;\
	    $(test-wrapper) cp $$dso $(objpfx)testroot.pristine$$dso ;\
	  done
endif
	# $(symbolic-link-list) is a file that encodes $(DESTDIR) so we
	# have to purge it
	rm -f $(symbolic-link-list)
	# Setting INSTALL_UNCOMPRESSED causes localedata/Makefile to
	# install the charmaps uncompressed, as the testroot does not
	# provide a gunzip program.
	$(MAKE) install DESTDIR=$(objpfx)testroot.pristine \
	  INSTALL_UNCOMPRESSED=yes subdirs='$(sorted-subdirs)'
	rm -f $(symbolic-link-list)
	touch $(objpfx)testroot.pristine/install.stamp

tests-special-notdir = $(patsubst $(objpfx)%, %, $(tests-special))
tests: $(tests-special)
	$(..)scripts/merge-test-results.sh -s $(objpfx) "" \
	  $(sort $(tests-special-notdir:.out=)) \
	  > $(objpfx)subdir-tests.sum
	$(..)scripts/merge-test-results.sh -t $(objpfx) subdir-tests.sum \
	  $(sort $(subdirs) .) \
	  > $(objpfx)tests.sum
	$(call summarize-tests,tests.sum)
xtests:
	$(..)scripts/merge-test-results.sh -t $(objpfx) subdir-xtests.sum \
	  $(sort $(subdirs)) \
	  > $(objpfx)xtests.sum
	$(call summarize-tests,xtests.sum, for extra tests)

# The realclean target is just like distclean for the parent, but we want
# the subdirs to know the difference in case they care.
realclean distclean: parent-clean
# This is done this way rather than having `subdir_distclean' be a
# dependency of this target so that libc.a will be removed before the
# subdirectories are dealt with and so they won't try to remove object
# files from it when it's going to be removed anyway.
	@$(MAKE) distclean-1 no_deps=t distclean-1=$@ avoid-generated=yes \
		 sysdep-subdirs="$(sysdep-subdirs)"
	-rm -f $(postclean)

# Subroutine of distclean and realclean.
distclean-1: subdir_$(distclean-1)
	-rm -f $(config-generated)
	-rm -f $(addprefix $(objpfx),config.status config.cache config.log)
	-rm -f $(addprefix $(objpfx),config.make config-name.h config.h)
ifdef objdir
	-rm -f $(objpfx)Makefile
endif
	-rm -f $(sysdep-$(distclean-1))

# Make the TAGS file for Emacs users.

.PHONY: TAGS
TAGS:
	scripts/list-sources.sh | sed -n -e '/Makefile/p' \
	  $(foreach S,[chsSyl] cxx sh bash pl,\
		    $(subst .,\.,-e '/.$S\(.in\)*$$/p')) \
	| $(ETAGS) -o $@ -

# Make the distribution tarfile.
.PHONY: dist dist-prepare

generated := $(generated) stubs.h

files-for-dist := README INSTALL configure NEWS

# Regenerate stuff, then error if these things are not committed yet.
dist-prepare: $(files-for-dist)
	conf=`find sysdeps -name configure`; \
	$(MAKE) $$conf && \
	git diff --stat HEAD -- $^ $$conf \
	| $(AWK) '{ print; rc=1 } END { exit rc }'

%.tar: FORCE
	git archive --prefix=$*/ $* > $@.new
	mv -f $@.new $@

# Do `make dist dist-version=X.Y.Z' to make tar files of an older version.

ifneq (,$(strip $(dist-version)))
dist: $(foreach Z,.bz2 .gz .xz,$(dist-version).tar$Z)
	md5sum $^
else
dist: dist-prepare
	@if v=`git describe`; then \
	  echo Distribution version $$v; \
	  $(MAKE) dist dist-version=$$v; \
	else \
	  false; \
	fi
endif

INSTALL: manual/install-plain.texi manual/macros.texi \
	 $(common-objpfx)manual/pkgvers.texi manual/install.texi
	makeinfo --no-validate --plaintext --no-number-sections \
		 -I$(common-objpfx)manual $< -o $@-tmp
	$(AWK) 'NF == 0 { ++n; next } \
		NF != 0 { while (n-- > 0) print ""; n = 0; print }' \
	  < $@-tmp > $@-tmp2
	rm -f $@-tmp
	-chmod a-w $@-tmp2
	mv -f $@-tmp2 $@
$(common-objpfx)manual/%: FORCE
	$(MAKE) $(PARALLELMFLAGS) -C manual $@
FORCE:

iconvdata/% localedata/% po/%: FORCE
	$(MAKE) $(PARALLELMFLAGS) -C $(@D) $(@F)

# Convenience target to rerun one test, from the top of the build tree
# Example: make test t=wcsmbs/test-wcsnlen
.PHONY: test
test :
	@-rm -f $(objpfx)$t.out
	$(MAKE) subdir=$(patsubst %/,%,$(dir $t)) -C $(dir $t) ..=../ $(objpfx)$t.out
	@cat $(objpfx)$t.test-result
	@cat $(objpfx)$t.out
