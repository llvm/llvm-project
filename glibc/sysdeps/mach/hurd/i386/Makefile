ifeq ($(subdir),misc)
sysdep_routines += ioperm
sysdep_headers += sys/io.h
endif

ifeq ($(subdir),debug)
gen-as-const-headers += signal-defines.sym
endif

ifeq ($(subdir),setjmp)
gen-as-const-headers += signal-defines.sym
endif

ifeq ($(subdir),csu)
ifeq (yes,$(build-shared))
sysdep_routines += divdi3
shared-only-routines += divdi3
CPPFLAGS-divdi3.c = -Din_divdi3_c
endif
endif

ifeq ($(subdir),conform)
# For bugs 23081, 23082, 23083, 23084, 23085, 23086.
# (type conformance)
conformtest-xfail-conds += i386-gnu

# For bug 23088
# (sched_* visibility)
test-xfail-POSIX/fcntl.h/conform = yes
test-xfail-POSIX/signal.h/conform = yes
test-xfail-POSIX/semaphore.h/conform = yes
test-xfail-POSIX/regex.h/conform = yes
test-xfail-POSIX/aio.h/conform = yes
test-xfail-POSIX/mqueue.h/conform = yes
test-xfail-POSIX/sys/types.h/conform = yes
test-xfail-UNIX98/fcntl.h/conform = yes
test-xfail-UNIX98/netdb.h/conform = yes
test-xfail-UNIX98/signal.h/conform = yes
test-xfail-UNIX98/semaphore.h/conform = yes
test-xfail-UNIX98/regex.h/conform = yes
test-xfail-UNIX98/aio.h/conform = yes
test-xfail-UNIX98/ftw.h/conform = yes
test-xfail-UNIX98/mqueue.h/conform = yes
test-xfail-UNIX98/netinet/in.h/conform = yes
test-xfail-UNIX98/sys/wait.h/conform = yes
test-xfail-UNIX98/sys/sem.h/conform = yes
test-xfail-UNIX98/sys/uio.h/conform = yes
test-xfail-UNIX98/sys/socket.h/conform = yes
test-xfail-UNIX98/sys/types.h/conform = yes
test-xfail-UNIX98/stdlib.h/conform = yes
test-xfail-UNIX98/arpa/inet.h/conform = yes
test-xfail-POSIX2008/fcntl.h/conform = yes
test-xfail-POSIX2008/netdb.h/conform = yes
test-xfail-POSIX2008/signal.h/conform = yes
test-xfail-POSIX2008/semaphore.h/conform = yes
test-xfail-POSIX2008/regex.h/conform = yes
test-xfail-POSIX2008/aio.h/conform = yes
test-xfail-POSIX2008/mqueue.h/conform = yes
test-xfail-POSIX2008/netinet/in.h/conform = yes
test-xfail-POSIX2008/sys/wait.h/conform = yes
test-xfail-POSIX2008/sys/socket.h/conform = yes
test-xfail-POSIX2008/sys/types.h/conform = yes
test-xfail-POSIX2008/arpa/inet.h/conform = yes
test-xfail-XOPEN2K/fcntl.h/conform = yes
test-xfail-XOPEN2K/netdb.h/conform = yes
test-xfail-XOPEN2K/signal.h/conform = yes
test-xfail-XOPEN2K/semaphore.h/conform = yes
test-xfail-XOPEN2K/regex.h/conform = yes
test-xfail-XOPEN2K/aio.h/conform = yes
test-xfail-XOPEN2K/ftw.h/conform = yes
test-xfail-XOPEN2K/mqueue.h/conform = yes
test-xfail-XOPEN2K/netinet/in.h/conform = yes
test-xfail-XOPEN2K/sys/wait.h/conform = yes
test-xfail-XOPEN2K/sys/sem.h/conform = yes
test-xfail-XOPEN2K/sys/uio.h/conform = yes
test-xfail-XOPEN2K/sys/socket.h/conform = yes
test-xfail-XOPEN2K/sys/types.h/conform = yes
test-xfail-XOPEN2K/stdlib.h/conform = yes
test-xfail-XOPEN2K/arpa/inet.h/conform = yes
test-xfail-XOPEN2K8/fcntl.h/conform = yes
test-xfail-XOPEN2K8/netdb.h/conform = yes
test-xfail-XOPEN2K8/signal.h/conform = yes
test-xfail-XOPEN2K8/semaphore.h/conform = yes
test-xfail-XOPEN2K8/regex.h/conform = yes
test-xfail-XOPEN2K8/aio.h/conform = yes
test-xfail-XOPEN2K8/ftw.h/conform = yes
test-xfail-XOPEN2K8/mqueue.h/conform = yes
test-xfail-XOPEN2K8/netinet/in.h/conform = yes
test-xfail-XOPEN2K8/sys/wait.h/conform = yes
test-xfail-XOPEN2K8/sys/sem.h/conform = yes
test-xfail-XOPEN2K8/sys/uio.h/conform = yes
test-xfail-XOPEN2K8/sys/socket.h/conform = yes
test-xfail-XOPEN2K8/sys/types.h/conform = yes
test-xfail-XOPEN2K8/stdlib.h/conform = yes
test-xfail-XOPEN2K8/arpa/inet.h/conform = yes
endif

# For bug 23286
ifeq ($(subdir),hurd)
test-xfail-check-abi-libhurduser = yes
endif
ifeq ($(subdir),mach)
test-xfail-check-abi-libmachuser = yes
endif

ifeq ($(subdir),htl)
# For bug 25521
# (setpshared support)
test-xfail-tst-mutex4 = yes
test-xfail-tst-cancel16 = yes
test-xfail-tst-cancelx16 = yes
test-xfail-tst-cond4 = yes
test-xfail-tst-cond6 = yes
test-xfail-tst-cond12 = yes
test-xfail-tst-cond13 = yes
test-xfail-tst-cond23 = yes
test-xfail-tst-rwlock4 = yes
test-xfail-tst-rwlock12 = yes
test-xfail-tst-barrier2 = yes
test-xfail-tst-pututxline-cache = yes
test-xfail-tst-pututxline-lockfail = yes
test-xfail-tst-mallocfork2 = yes
test-xfail-tst-flock2 = yes
test-xfail-tst-signal1 = yes
test-xfail-tst-signal2 = yes

# For bug 25522
# (setprotocol support)
test-xfail-tst-cond24 = yes
test-xfail-tst-cond25 = yes

# For bug 25563
# (robust support against dead threads)
test-xfail-tst-robust1 = yes
test-xfail-tst-robust2 = yes
test-xfail-tst-robust3 = yes
test-xfail-tst-robust4 = yes
test-xfail-tst-robust5 = yes
test-xfail-tst-robust6 = yes
test-xfail-tst-robust7 = yes
test-xfail-tst-robust9 = yes
endif

ifeq ($(subdir),elf)
# We do use nested functions involving creation of trampolines, notably for
# callbacks whose parameters don't permit to get the context parameters.
check-execstack-xfail += ld.so libc.so libpthread.so
# We always create a thread for signals
test-xfail-tst-single_threaded-pthread-static = yes
endif

ifeq ($(subdir),stdlib)
gen-as-const-headers += ucontext_i.sym
endif
