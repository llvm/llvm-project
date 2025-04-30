# Copyright (C) 1999-2021 Free Software Foundation, Inc.
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
#	Sub-makefile for conform portion of the library.
#
subdir	:= conform

include ../Makeconfig

conformtest-headers-data := $(wildcard data/*.h-data) \
			    $(wildcard data/*/*.h-data)

conformtest-standards := ISO ISO99 ISO11 POSIX XPG4 XPG42 UNIX98 XOPEN2K \
			 POSIX2008 XOPEN2K8

conformtest-headers-ISO := assert.h ctype.h errno.h float.h limits.h locale.h \
			   math.h setjmp.h signal.h stdarg.h stddef.h stdio.h \
			   stdlib.h string.h time.h
conformtest-headers-ISO99 := $(conformtest-headers-ISO) complex.h fenv.h \
			     inttypes.h iso646.h stdbool.h stdint.h tgmath.h \
			     wchar.h wctype.h
# Missing ISO11 expectations for: stdatomic.h
conformtest-headers-ISO11 := $(conformtest-headers-ISO99) stdalign.h \
			     stdnoreturn.h threads.h uchar.h
conformtest-headers-POSIX := $(conformtest-headers-ISO) aio.h dirent.h \
			     fcntl.h fnmatch.h glob.h grp.h mqueue.h \
			     pthread.h pwd.h regex.h sched.h semaphore.h \
			     sys/mman.h sys/stat.h sys/times.h sys/types.h \
			     sys/utsname.h sys/wait.h tar.h termios.h \
			     unistd.h utime.h wordexp.h
# Missing XPG4 expectations for: regexp.h wchar.h.
conformtest-headers-XPG4 := $(conformtest-headers-ISO) cpio.h dirent.h \
			    fcntl.h fnmatch.h ftw.h glob.h grp.h iconv.h \
			    langinfo.h monetary.h nl_types.h pwd.h regex.h \
			    search.h sys/ipc.h sys/msg.h sys/sem.h sys/shm.h \
			    sys/stat.h sys/times.h sys/types.h sys/utsname.h \
			    sys/wait.h tar.h termios.h ulimit.h unistd.h \
			    utime.h varargs.h wordexp.h
# Missing XPG42 expectations for: re_comp.h regexp.h wchar.h.
# XPG42 includes XTI and STREAMS, but those are not implemented by glibc.
conformtest-headers-XPG42 := $(conformtest-headers-XPG4) arpa/inet.h fmtmsg.h \
			    libgen.h ndbm.h netdb.h netinet/in.h poll.h \
			    strings.h sys/mman.h sys/resource.h \
			    sys/socket.h sys/statvfs.h sys/time.h sys/timeb.h \
			    sys/uio.h sys/un.h syslog.h ucontext.h utmpx.h
# Missing UNIX98 expectations for: inttypes.h re_comp.h regexp.h.
# The online UNIX98 includes XCURSES, but curses.h, term.h and
# unctrl.h are outside the scope of these tests.  It also includes
# XTI and STREAMS, but those are not implemented by glibc.
conformtest-headers-UNIX98 := $(conformtest-headers-POSIX) arpa/inet.h cpio.h \
			      dlfcn.h fmtmsg.h ftw.h iconv.h iso646.h \
			      langinfo.h libgen.h monetary.h ndbm.h netdb.h \
			      netinet/in.h nl_types.h poll.h search.h \
			      strings.h sys/ipc.h sys/msg.h \
			      sys/resource.h sys/sem.h sys/shm.h sys/socket.h \
			      sys/statvfs.h sys/time.h sys/timeb.h sys/uio.h \
			      sys/un.h syslog.h ucontext.h ulimit.h utmpx.h \
			      varargs.h wchar.h wctype.h
# Missing XOPEN2K expectations for: trace.h, stropts.h.
conformtest-headers-XOPEN2K := $(conformtest-headers-POSIX) arpa/inet.h \
			       complex.h cpio.h dlfcn.h fenv.h fmtmsg.h ftw.h \
			       iconv.h inttypes.h iso646.h langinfo.h \
			       libgen.h monetary.h ndbm.h net/if.h netdb.h \
			       netinet/in.h netinet/tcp.h nl_types.h poll.h \
			       search.h spawn.h stdbool.h stdint.h strings.h \
			       sys/ipc.h sys/msg.h sys/resource.h \
			       sys/select.h sys/sem.h sys/shm.h sys/socket.h \
			       sys/statvfs.h sys/time.h sys/timeb.h sys/uio.h \
			       sys/un.h syslog.h tgmath.h ucontext.h ulimit.h \
			       utmpx.h wchar.h wctype.h
# Missing POSIX2008 expectations for: trace.h, stropts.h.
conformtest-headers-POSIX2008 := $(conformtest-headers-POSIX) arpa/inet.h \
				 complex.h cpio.h dlfcn.h fenv.h iconv.h \
				 inttypes.h iso646.h langinfo.h monetary.h \
				 net/if.h netdb.h netinet/in.h netinet/tcp.h \
				 nl_types.h poll.h spawn.h stdbool.h stdint.h \
				 strings.h sys/select.h \
				 sys/socket.h sys/statvfs.h sys/un.h tgmath.h \
				 wchar.h wctype.h
# Missing XOPEN2K8 expectations for: trace.h.
conformtest-headers-XOPEN2K8 := $(conformtest-headers-POSIX2008) fmtmsg.h \
				ftw.h libgen.h ndbm.h search.h sys/ipc.h \
				sys/msg.h sys/resource.h sys/sem.h sys/shm.h \
				sys/time.h sys/uio.h syslog.h ulimit.h utmpx.h

conformtest-header-list-base := $(foreach std,$(conformtest-standards),\
					      header-list-$(std).out)
conformtest-header-list-tests := $(addprefix $(objpfx),\
					     $(conformtest-header-list-base))
tests-special += $(conformtest-header-list-tests)
generated += $(conformtest-header-list-base)

conformtest-header-base := $(foreach std,\
				     $(conformtest-standards),\
				     $(foreach h,\
					       $(conformtest-headers-$(std)),\
					       $(std)/$(h)/conform.out))
conformtest-header-tests := $(addprefix $(objpfx),$(conformtest-header-base))
ifneq (yes,$(fast-check))
tests-special += $(conformtest-header-tests)
generated += $(conformtest-header-base)
endif

linknamespace-symlists-base := $(foreach std,$(conformtest-standards),\
					     symlist-$(std))
linknamespace-symlists-tests := $(addprefix $(objpfx),\
					    $(linknamespace-symlists-base))
tests-special += $(linknamespace-symlists-tests)

linknamespace-symlist-stdlibs-base := $(foreach std,$(conformtest-standards),\
						    symlist-stdlibs-$(std))
linknamespace-symlist-stdlibs-tests := \
	$(addprefix $(objpfx),\
		    $(linknamespace-symlist-stdlibs-base))

tests-special += $(linknamespace-symlist-stdlibs-tests)

linknamespace-header-base := $(foreach std,\
				       $(conformtest-standards),\
				       $(foreach h,\
						 $(conformtest-headers-$(std)),\
						 $(std)/$(h)/linknamespace.out))
linknamespace-header-tests := $(addprefix $(objpfx),\
					  $(linknamespace-header-base))
tests-special += $(linknamespace-header-tests)

include ../Rules

$(conformtest-header-list-tests): $(objpfx)header-list-%.out: \
				  check-header-lists.sh \
				  $(conformtest-headers-data)
	$(SHELL) $< "$*" "$(CC)" "$(strip $(conformtest-headers-$*))" \
		 "$(conformtest-headers-data)" > $@; \
	$(evaluate-test)

# Pre-standard C feature no longer supported by GCC (obsoleted in
# newer POSIX standards).
test-xfail-XPG4/varargs.h/conform = yes
test-xfail-XPG42/varargs.h/conform = yes
test-xfail-UNIX98/varargs.h/conform = yes

# Header not provided by glibc.
test-xfail-XPG42/ndbm.h/conform = yes
test-xfail-UNIX98/ndbm.h/conform = yes
test-xfail-XOPEN2K/ndbm.h/conform = yes
test-xfail-XOPEN2K8/ndbm.h/conform = yes

conformtest-cc-flags = -I../include $(+sysdep-includes) $(sysincludes) -I..
# conformtest-xfail-conds may be set by a sysdeps Makefile fragment to
# a list of conditions that are considered to be true when encountered
# in xfail[cond]- lines in test expectations.
conformtest-xfail = $(if $(conformtest-xfail-conds),\
			 --xfail='$(conformtest-xfail-conds)')
ifeq (no,$(cross-compiling))
conformtest-cross =
else
conformtest-cross = --cross
endif
$(conformtest-header-tests): $(objpfx)%/conform.out: \
			     conformtest.py $(conformtest-headers-data)
	(set -e; std_hdr=$*; std=$${std_hdr%%/*}; hdr=$${std_hdr#*/}; \
	 mkdir -p $(@D); \
	 $(PYTHON) $< --cc='$(CC)' --flags='$(conformtest-cc-flags)' \
		   --standard=$$std --header=$$hdr $(conformtest-xfail) \
		   $(conformtest-cross) \
		   > $@ 2>&1); \
	$(evaluate-test)

$(linknamespace-symlists-tests): $(objpfx)symlist-%: list-header-symbols.py
	$(PYTHON) $< --cc='$(CC)' --flags='$(conformtest-cc-flags)' \
		  --standard=$* --headers="$(strip $(conformtest-headers-$*))" \
		  > $@ 2> $@.err; \
	$(evaluate-test)

linknamespace-libs-isoc = $(common-objpfx)libc.a $(common-objpfx)math/libm.a
linknamespace-libs-thr = $(linknamespace-libs-isoc) \
			 $(common-objpfx)rt/librt.a \
			 $(filter-out %_syms.a,$(static-thread-library))
linknamespace-libs-posix = $(linknamespace-libs-thr) \
			   $(common-objpfx)dlfcn/libdl.a
linknamespace-libs-xsi = $(linknamespace-libs-posix)
linknamespace-libs-ISO = $(linknamespace-libs-isoc)
linknamespace-libs-ISO99 = $(linknamespace-libs-isoc)
linknamespace-libs-ISO11 = $(linknamespace-libs-isoc) \
			   $(filter-out %_syms.a,$(static-thread-library))
linknamespace-libs-XPG4 = $(linknamespace-libs-isoc)
linknamespace-libs-XPG42 = $(linknamespace-libs-XPG4)
linknamespace-libs-POSIX = $(linknamespace-libs-thr)
linknamespace-libs-UNIX98 = $(linknamespace-libs-xsi)
linknamespace-libs-XOPEN2K = $(linknamespace-libs-xsi)
linknamespace-libs-POSIX2008 = $(linknamespace-libs-posix)
linknamespace-libs-XOPEN2K8 = $(linknamespace-libs-xsi)
linknamespace-libs = $(foreach std,$(conformtest-standards),\
				   $(linknamespace-libs-$(std)))

ifeq ($(build-crypt),yes)
linknamespace-libs-xsi += $(common-objpfx)crypt/libcrypt.a
linknamespace-libs-XPG4 += $(common-objpfx)crypt/libcrypt.a
endif

$(linknamespace-symlist-stdlibs-tests): $(objpfx)symlist-stdlibs-%: \
					$(linknamespace-libs)
	LC_ALL=C $(READELF) -W -s $(linknamespace-libs-$*) > $@; \
	$(evaluate-test)

$(linknamespace-header-tests): $(objpfx)%/linknamespace.out: \
			       linknamespace.py \
			       $(linknamespace-symlists-tests) \
			       $(linknamespace-symlist-stdlibs-tests)
	(set -e; std_hdr=$*; std=$${std_hdr%%/*}; hdr=$${std_hdr#*/}; \
	 mkdir -p $(@D); \
	 $(PYTHON) $< --cc='$(CC)' --flags='$(conformtest-cc-flags)' \
		   --standard=$$std --stdsyms=$(objpfx)symlist-$$std \
		   --header=$$hdr --libsyms=$(objpfx)symlist-stdlibs-$$std \
		   --readelf='$(READELF)' \
		   > $@ 2>&1); \
	$(evaluate-test)

# Pre-standard C feature no longer supported by GCC (obsoleted in
# newer POSIX standards).
test-xfail-XPG4/varargs.h/linknamespace = yes
test-xfail-XPG42/varargs.h/linknamespace = yes
test-xfail-UNIX98/varargs.h/linknamespace = yes

# Header not provided by glibc.
test-xfail-XPG42/ndbm.h/linknamespace = yes
test-xfail-UNIX98/ndbm.h/linknamespace = yes
test-xfail-XOPEN2K/ndbm.h/linknamespace = yes
test-xfail-XOPEN2K8/ndbm.h/linknamespace = yes
