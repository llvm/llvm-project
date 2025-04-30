# Option to pass to Python scripts to set the C compiler.  Rewriting
# MODULE_NAME is required to enable the _ISOMAC verbatim header
# environment.
sysdeps-linux-python-cc = \
  --cc="$(CC) $(patsubst -DMODULE_NAME=%,-DMODULE_NAME=testsuite,$(CPPFLAGS))"

# Additional dependencies for Python scripts.
sysdeps-linux-python-deps = \
  $(..)sysdeps/unix/sysv/linux/glibcsyscalls.py \
  $(..)scripts/glibcextract.py

# Invocation of the Python interpreter with the Python search path.
sysdeps-linux-python = \
  PYTHONPATH=$(..)scripts:$(..)sysdeps/unix/sysv/linux $(PYTHON)

ifndef subdir
# This target performs two actions:
#
# Replace <arch-syscall.h> with a file generated from kernel headers
# and <fixup-asm-unistd.h>.  Both files are located via the sysdeps
# override search path.
#
# Update sysdeps/unix/sysv/linux/syscall-names.list with additional
# names found in the generated <arch-syscall.h> file, so that the
# global system call names list is a superset of the
# architecture-specific system call names.
#
# To bootstrap a new architecture, create an empty file in the right
# place and run `make update-syscall-lists' from the top-level of a
# configured, but not-yet-built glibc tree.
#
# --lock points to a file not replaced during the update operation, so
# that mutual exclusion is achieved.
.PHONY: update-syscall-lists
update-syscall-lists: arch-syscall.h
	$(sysdeps-linux-python) \
	  sysdeps/unix/sysv/linux/update-syscall-lists.py \
	    $(sysdeps-linux-python-cc) \
	    --lock=sysdeps/unix/sysv/linux/update-syscall-lists.py $< \
	    sysdeps/unix/sysv/linux/syscall-names.list
endif

ifeq ($(subdir),csu)
sysdep_routines += errno-loc
endif

ifeq ($(subdir),assert)
CFLAGS-assert.c += -DFATAL_PREPARE_INCLUDE='<fatal-prepare.h>'
CFLAGS-assert-perr.c += -DFATAL_PREPARE_INCLUDE='<fatal-prepare.h>'
endif

ifeq ($(subdir),malloc)
CFLAGS-malloc.c += -DMORECORE_CLEARS=2
endif

ifeq ($(subdir),misc)
sysdep_routines += adjtimex clone umount umount2 readahead sysctl \
		   setfsuid setfsgid epoll_pwait signalfd \
		   eventfd eventfd_read eventfd_write prlimit \
		   personality epoll_wait tee vmsplice splice \
		   open_by_handle_at mlock2 pkey_mprotect pkey_set pkey_get \
		   timerfd_gettime timerfd_settime prctl \
		   process_vm_readv process_vm_writev clock_adjtime \
		   pselect32 \
		   xstat fxstat lxstat xstat64 fxstat64 lxstat64 \
		   fxstatat fxstatat64 \
		   xmknod xmknodat convert_scm_timestamps \
		   closefrom_fallback \
		   clone3 clone-internal

CFLAGS-gethostid.c = -fexceptions
CFLAGS-tee.c = -fexceptions -fasynchronous-unwind-tables
CFLAGS-vmsplice.c = -fexceptions -fasynchronous-unwind-tables
CFLAGS-splice.c = -fexceptions -fasynchronous-unwind-tables
CFLAGS-open_by_handle_at.c = -fexceptions -fasynchronous-unwind-tables
CFLAGS-sync_file_range.c = -fexceptions -fasynchronous-unwind-tables
CFLAGS-pselect32.c = -fexceptions -fasynchronous-unwind-tables
CFLAGS-tst-writev.c += "-DARTIFICIAL_LIMIT=(0x80000000-sysconf(_SC_PAGESIZE))"

sysdep_headers += sys/mount.h sys/acct.h \
		  sys/klog.h \
		  sys/user.h sys/prctl.h \
		  sys/kd.h sys/soundcard.h sys/vt.h \
		  sys/quota.h sys/fsuid.h \
		  scsi/sg.h scsi/scsi.h scsi/scsi_ioctl.h sys/pci.h \
		  sys/raw.h sys/personality.h sys/epoll.h \
		  bits/a.out.h sys/inotify.h sys/signalfd.h sys/eventfd.h \
		  sys/timerfd.h sys/fanotify.h bits/eventfd.h bits/inotify.h \
		  bits/signalfd.h bits/timerfd.h bits/epoll.h \
		  bits/socket_type.h bits/syscall.h \
		  bits/mman-linux.h bits/mman-shared.h bits/ptrace-shared.h \
		  bits/siginfo-arch.h bits/siginfo-consts-arch.h \
		  bits/procfs.h bits/procfs-id.h bits/procfs-extra.h \
		  bits/procfs-prregset.h bits/mman-map-flags-generic.h \
		  bits/shmlba.h \
		  bits/termios-struct.h bits/termios-c_cc.h \
		  bits/termios-c_iflag.h bits/termios-c_oflag.h \
		  bits/termios-baud.h bits/termios-c_cflag.h \
		  bits/termios-c_lflag.h bits/termios-tcflow.h \
		  bits/termios-misc.h \
		  bits/types/struct_semid_ds.h \
		  bits/types/struct_msqid_ds.h \
		  bits/types/struct_shmid_ds.h \
		  bits/ipc-perm.h \
		  bits/struct_stat.h \
		  bits/struct_stat_time64_helper.h \
		  bits/types/struct_msqid64_ds.h \
		  bits/types/struct_msqid64_ds_helper.h \
		  bits/types/struct_semid64_ds.h \
		  bits/types/struct_semid64_ds_helper.h \
		  bits/types/struct_shmid64_ds.h \
		  bits/types/struct_shmid64_ds_helper.h \
		  bits/pthread_stack_min.h bits/pthread_stack_min-dynamic.h

tests += tst-clone tst-clone2 tst-clone3 tst-fanotify tst-personality \
	 tst-quota tst-sync_file_range tst-sysconf-iov_max tst-ttyname \
	 test-errno-linux tst-memfd_create tst-mlock2 tst-pkey \
	 tst-rlimit-infinity tst-ofdlocks tst-gettid tst-gettid-kill \
	 tst-tgkill tst-sysvsem-linux tst-sysvmsg-linux tst-sysvshm-linux \
	 tst-timerfd tst-ppoll \
	 tst-clock_adjtime tst-adjtimex tst-ntp_adjtime tst-ntp_gettime \
	 tst-ntp_gettimex tst-sigtimedwait tst-misalign-clone \
  tst-close_range \
  tst-prctl \
  tst-scm_rights \
  # tests

# Test for the symbol version of fcntl that was replaced in glibc 2.28.
ifeq ($(have-GLIBC_2.27)$(build-shared),yesyes)
tests += tst-ofdlocks-compat
endif

tests-internal += tst-sigcontext-get_pc

tests-time64 += \
  tst-adjtimex-time64 \
  tst-clock_adjtime-time64 \
  tst-ntp_adjtime-time64 \
  tst-ntp_gettime-time64 \
  tst-ntp_gettimex-time64 \
  tst-ppoll-time64 \
  tst-sigtimedwait-time64 \
  tst-timerfd-time64 \
  tst-prctl-time64 \
  tst-scm_rights-time64 \
  # tests-time64

tests-clone-internal = \
  tst-align-clone-internal \
  tst-clone2-internal \
  tst-clone3-internal \
  tst-getpid1-internal \
  tst-misalign-clone-internal
tests-internal += $(tests-clone-internal)
tests-static += $(tests-clone-internal)

CFLAGS-tst-sigcontext-get_pc.c = -fasynchronous-unwind-tables

# Generate the list of SYS_* macros for the system calls (__NR_*
# macros).  The file syscall-names.list contains all possible system
# call names, and the generated header file produces SYS_* macros for
# the __NR_* macros which are actually defined.

generated += bits/syscall.h
$(objpfx)bits/syscall.h: \
  ../sysdeps/unix/sysv/linux/gen-syscall-h.awk \
  ../sysdeps/unix/sysv/linux/syscall-names.list
	$(make-target-directory)
	LC_ALL=C $(AWK) -f $^ > $@-tmp
	$(move-if-change) $@-tmp $@
before-compile += $(objpfx)bits/syscall.h

# All macros defined by <sys/syscall.h>.  Include <bits/syscall.h>
# explicitly because <sys/sycall.h> skips it if _LIBC is defined.
$(objpfx)tst-syscall-list-macros.list: \
  $(objpfx)bits/syscall.h ../sysdeps/unix/sysv/linux/sys/syscall.h
	printf '#include <linux/version.h>\n\
#include <sys/syscall.h>\n#include <bits/syscall.h>\n' | \
	  $(CC) -E -o $@-tmp $(CFLAGS) $(CPPFLAGS) -x c - -dM
	$(move-if-change) $@-tmp $@

# __NR_* system call names.  Used by the test below.
$(objpfx)tst-syscall-list-nr.list: \
  ../sysdeps/unix/sysv/linux/filter-nr-syscalls.awk \
  $(objpfx)tst-syscall-list-macros.list
	LC_ALL=C $(AWK) -f $^ > $@-tmp
	$(move-if-change) $@-tmp $@

# SYS_* system call names.  Used by the test below.
$(objpfx)tst-syscall-list-sys.list: $(objpfx)tst-syscall-list-macros.list
	LC_ALL=C $(AWK) '/^#define SYS_/ { print substr($$2, 5) }' $< > $@-tmp
	$(move-if-change) $@-tmp $@

tests-special += $(objpfx)tst-syscall-list.out
$(objpfx)tst-syscall-list.out: \
  ../sysdeps/unix/sysv/linux/tst-syscall-list.sh \
  $(objpfx)tst-syscall-list-macros.list \
  $(objpfx)tst-syscall-list-nr.list \
  $(objpfx)tst-syscall-list-sys.list
	$(BASH) $^ $(AWK) > $@; $(evaluate-test)

tests-special += $(objpfx)tst-glibcsyscalls.out
# arch-syscall.h is located via the sysdeps override search path.
$(objpfx)tst-glibcsyscalls.out: arch-syscall.h \
  ../sysdeps/unix/sysv/linux/syscall-names.list
	$(sysdeps-linux-python) \
	  ../sysdeps/unix/sysv/linux/tst-glibcsyscalls.py \
	    $(sysdeps-linux-python-cc) $< \
	    ../sysdeps/unix/sysv/linux/syscall-names.list \
	  < /dev/null > $@ 2>&1; $(evaluate-test)
$(objpfx)tst-glibcsyscalls.out: \
  ../sysdeps/unix/sysv/linux/tst-glibcsyscalls.py \
  $(sysdeps-linux-python-deps)

# Separate object file for access to the constant from the UAPI header.
$(objpfx)tst-sysconf-iov_max: $(objpfx)tst-sysconf-iov_max-uapi.o

tests-special += $(objpfx)tst-mman-consts.out
$(objpfx)tst-mman-consts.out: ../sysdeps/unix/sysv/linux/tst-mman-consts.py
	$(sysdeps-linux-python) \
	  ../sysdeps/unix/sysv/linux/tst-mman-consts.py \
	    $(sysdeps-linux-python-cc) \
	  < /dev/null > $@ 2>&1; $(evaluate-test)
$(objpfx)tst-mman-consts.out: $(sysdeps-linux-python-deps)

endif # $(subdir) == misc

ifeq ($(subdir),time)
sysdep_headers += sys/timex.h bits/timex.h

sysdep_routines += ntp_gettime ntp_gettimex

tests += \
  tst-clock_gettime-clobber \
  tst-gettimeofday-clobber \
  tst-time-clobber \
  # tests
endif

ifeq ($(subdir),signal)
tests-special += $(objpfx)tst-signal-numbers.out
# Depending on signal.o* is a hack.  What we actually want is a dependency
# on signal.h and everything it includes.  That's impractical to write
# in this context, but signal.c includes signal.h and not much else so it'll
# be conservatively correct.
$(objpfx)tst-signal-numbers.out: \
		../sysdeps/unix/sysv/linux/tst-signal-numbers.py \
		$(objpfx)signal.o*
	$(sysdeps-linux-python) \
	  ../sysdeps/unix/sysv/linux/tst-signal-numbers.py \
	    $(sysdeps-linux-python-cc) \
	  < /dev/null > $@ 2>&1; $(evaluate-test)
$(objpfx)tst-signal-numbers.out: $(sysdeps-linux-python-deps)
endif

ifeq ($(subdir),socket)
sysdep_headers += net/if_ppp.h net/ppp-comp.h \
		  net/ppp_defs.h net/if_arp.h net/route.h net/ethernet.h \
		  net/if_slip.h net/if_packet.h net/if_shaper.h \
		  bits/socket-constants.h
sysdep_routines += cmsg_nxthdr
CFLAGS-recvmmsg.c = -fexceptions -fasynchronous-unwind-tables
CFLAGS-sendmmsg.c = -fexceptions -fasynchronous-unwind-tables

tests-special += $(objpfx)tst-socket-consts.out
$(objpfx)tst-socket-consts.out: ../sysdeps/unix/sysv/linux/tst-socket-consts.py
	PYTHONPATH=../scripts \
	$(PYTHON) ../sysdeps/unix/sysv/linux/tst-socket-consts.py \
		   --cc="$(CC) $(patsubst -DMODULE_NAME=%, \
					  -DMODULE_NAME=testsuite, \
					  $(CPPFLAGS)) -D_ISOMAC" \
	< /dev/null > $@ 2>&1; $(evaluate-test)
endif # $(subdir) == socket

ifeq ($(subdir),sunrpc)
sysdep_headers += nfs/nfs.h
endif

ifeq ($(subdir),termios)
sysdep_headers += termio.h
endif

ifeq ($(subdir),posix)
sysdep_headers += bits/initspin.h

sysdep_routines += sched_getcpu oldglob getcpu

tests += tst-affinity tst-affinity-pid

tests-static := tst-affinity-static
tests += $(tests-static)

CFLAGS-fork.c = $(libio-mtsafe)
CFLAGS-getpid.o = -fomit-frame-pointer
CFLAGS-getpid.os = -fomit-frame-pointer
endif

ifeq ($(subdir),inet)
sysdep_headers += netinet/if_fddi.h netinet/if_tr.h \
		  netipx/ipx.h netash/ash.h netax25/ax25.h netatalk/at.h \
		  netrom/netrom.h netpacket/packet.h netrose/rose.h \
		  neteconet/ec.h netiucv/iucv.h
sysdep_routines += netlink_assert_response
endif

# Don't compile the ctype glue code, since there is no old non-GNU C library.
inhibit-glue = yes

ifeq ($(subdir),dirent)
sysdep_routines += getdirentries getdirentries64
tests += tst-getdents64

# The tested readdir64 symbol was replaced in glibc 2.2.
ifeq ($(have-GLIBC_2.1.3)$(build-shared),yesyes)
tests += tst-readdir64-compat
endif
endif # $(subdir) == dirent

ifeq ($(subdir),nis)
CFLAGS-ypclnt.c = -DUSE_BINDINGDIR=1
endif

ifeq ($(subdir),io)
sysdep_routines += xstatconv internal_statvfs \
		   sync_file_range fallocate fallocate64 \
		   close_nocancel fcntl_nocancel \
		   open_nocancel open64_nocancel \
		   openat_nocancel openat64_nocancel \
		   read_nocancel pread64_nocancel \
		   write_nocancel statx_cp stat_t64_cp

sysdep_headers += bits/fcntl-linux.h

tests += tst-fallocate tst-fallocate64 tst-o_path-locks
endif

ifeq ($(subdir),elf)
sysdep-rtld-routines += dl-brk dl-sbrk dl-getcwd dl-openat64 dl-opendir

libof-lddlibc4 = lddlibc4

others += pldd
install-bin += pldd
$(objpfx)pldd: $(objpfx)xmalloc.o
endif

ifeq ($(subdir),rt)
CFLAGS-mq_send.c += -fexceptions
CFLAGS-mq_receive.c += -fexceptions
endif

ifeq ($(subdir),nscd)
sysdep-CFLAGS += -DHAVE_EPOLL -DHAVE_INOTIFY -DHAVE_NETLINK
CFLAGS-gai.c += -DNEED_NETLINK
endif

ifeq ($(subdir),nptl)
tests += tst-align-clone tst-getpid1
endif
