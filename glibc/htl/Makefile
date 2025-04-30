#
#   Copyright (C) 1994-2021 Free Software Foundation, Inc.
#
#   The GNU C Library is free software; you can redistribute it and/or
#   modify it under the terms of the GNU Lesser General Public
#   License as published by the Free Software Foundation; either
#   version 2.1 of the License, or (at your option) any later version.

#   The GNU C Library is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   Lesser General Public License for more details.

#   You should have received a copy of the GNU Lesser General Public
#   License along with the GNU C Library; if not, see
#   <https://www.gnu.org/licenses/>.

subdir := htl

srcdir = .

MICROKERNEL := mach
SYSDEPS := lockfile

LCLHDRS :=

libpthread-routines := pt-attr pt-attr-destroy pt-attr-getdetachstate	    \
	pt-attr-getguardsize pt-attr-getinheritsched			    \
	pt-attr-getschedparam pt-attr-getschedpolicy pt-attr-getscope	    \
	pt-attr-getstack pt-attr-getstackaddr pt-attr-getstacksize	    \
	pt-attr-init pt-attr-setdetachstate pt-attr-setguardsize	    \
	pt-attr-setinheritsched pt-attr-setschedparam			    \
	pt-attr-setschedpolicy pt-attr-setscope pt-attr-setstack	    \
	pt-attr-setstackaddr pt-attr-setstacksize			    \
									    \
	pt-barrier-destroy pt-barrier-init pt-barrier-wait		    \
	pt-barrier pt-barrierattr-destroy pt-barrierattr-init		    \
	pt-barrierattr-getpshared pt-barrierattr-setpshared		    \
									    \
	pt-destroy-specific pt-init-specific				    \
	pt-key-create pt-key-delete					    \
	pt-getspecific pt-setspecific					    \
									    \
	pt-once								    \
									    \
	pt-alloc							    \
	pt-create							    \
	pt-getattr							    \
	pt-equal							    \
	pt-dealloc							    \
	pt-detach							    \
	pt-exit								    \
	pt-initialize							    \
	pt-join								    \
	pt-self								    \
	pt-sigmask							    \
	pt-spin-inlines							    \
	pt-cleanup							    \
	pt-setcancelstate						    \
	pt-setcanceltype						    \
	pt-testcancel							    \
	pt-cancel							    \
									    \
	pt-mutexattr-destroy pt-mutexattr-init				    \
	pt-mutexattr-getprioceiling pt-mutexattr-getprotocol		    \
	pt-mutexattr-getpshared pt-mutexattr-gettype			    \
	pt-mutexattr-setprioceiling pt-mutexattr-setprotocol		    \
	pt-mutexattr-setpshared pt-mutexattr-settype			    \
	pt-mutexattr-getrobust pt-mutexattr-setrobust			    \
									    \
	pt-mutex-init pt-mutex-destroy pt-mutex-checklocked		    \
	pt-mutex-lock pt-mutex-trylock pt-mutex-timedlock		    \
	pt-mutex-unlock							    \
	pt-mutex-transfer-np						    \
	pt-mutex-getprioceiling pt-mutex-setprioceiling			    \
	pt-mutex-consistent 						    \
									    \
	pt-rwlock-attr							    \
	pt-rwlockattr-init pt-rwlockattr-destroy			    \
	pt-rwlockattr-getpshared pt-rwlockattr-setpshared		    \
									    \
	pt-rwlock-init pt-rwlock-destroy				    \
	pt-rwlock-rdlock pt-rwlock-tryrdlock				    \
	pt-rwlock-trywrlock pt-rwlock-wrlock				    \
	pt-rwlock-timedrdlock pt-rwlock-timedwrlock			    \
	pt-rwlock-unlock						    \
									    \
	pt-cond								    \
	pt-condattr-init pt-condattr-destroy				    \
	pt-condattr-getclock pt-condattr-getpshared			    \
	pt-condattr-setclock pt-condattr-setpshared			    \
									    \
	pt-cond-destroy pt-cond-init					    \
	pt-cond-brdcast							    \
	pt-cond-signal							    \
	pt-cond-wait							    \
	pt-cond-timedwait						    \
	pt-hurd-cond-wait						    \
	pt-hurd-cond-timedwait						    \
									    \
	pt-stack-alloc							    \
	pt-thread-alloc							    \
	pt-thread-start							    \
	pt-thread-terminate						    \
	pt-startup							    \
									    \
	pt-getconcurrency pt-setconcurrency				    \
									    \
	pt-block							    \
	pt-timedblock							    \
	pt-block-intr							    \
	pt-timedblock-intr						    \
	pt-wakeup							    \
	pt-docancel							    \
	pt-sysdep							    \
	pt-setup							    \
	pt-machdep							    \
	pt-spin								    \
									    \
	pt-sigstate-init						    \
	pt-sigstate-destroy						    \
	pt-sigstate							    \
									    \
	pt-kill								    \
	pt-getcpuclockid						    \
									    \
	pt-getschedparam pt-setschedparam pt-setschedprio		    \
	pt-yield							    \
									    \
	sem_close sem-destroy sem-getvalue sem-init sem_open		    \
	sem-post sem-timedwait sem-trywait sem_unlink			    \
	sem-wait sem-waitfast						    \
	sem_routines							    \
									    \
	cancellation							    \
	cthreads-compat							    \
	herrno								    \
	$(SYSDEPS)

headers :=				\
              pthread.h				\
	      semaphore.h			\
						\
              bits/pthread.h			\
              bits/pthread-np.h			\
              bits/pthreadtypes.h		\
              bits/pthreadtypes-arch.h		\
              bits/thread-shared-types.h	\
              bits/types/struct___pthread_mutex.h	\
              bits/types/struct___pthread_cond.h	\
              bits/types/struct___pthread_condattr.h	\
              bits/types/__pthread_spinlock_t.h	\
              bits/spin-lock-inline.h		\
              bits/cancelation.h		\
              bits/types/struct___pthread_attr.h	\
              bits/types/struct___pthread_barrierattr.h	\
              bits/types/struct___pthread_barrier.h	\
              bits/types/__pthread_key.h		\
              bits/types/struct___pthread_once.h	\
              bits/types/struct___pthread_mutexattr.h	\
              bits/types/struct___pthread_rwlock.h	\
              bits/types/struct___pthread_rwlockattr.h	\
	      bits/semaphore.h

distribute :=

routines := forward libc_pthread_init alloca_cutoff htlfreeres
shared-only-routines = forward

extra-libs := libpthread
extra-libs-others := $(extra-libs)
install-lib := libpthread.so
install-lib-ldscripts := libpthread_syms.a

include ../Makeconfig

CFLAGS-lockfile.c = -D_IO_MTSAFE_IO

all: # Make this the default target; it will be defined in Rules.

subdir_install: $(inst_libdir)/libpthread2.a $(inst_libdir)/libpthread_syms.a

# XXX: If $(inst_libdir)/libpthread2.a is installed and
# $(inst_libdir)/libpthread is not, we can have some issues.
.PHONY: $(inst_libdir)/libpthread.a $(inst_libdir)/libpthread_pic.a

# XXX: These rules are a hack.  But it is better than messing with
# ../Makeconf at the moment.  Note that the linker scripts
# $(srcdir)/libpthread.a and $(srcdir)/libpthread_pic.a get overwritten
# when building in $(srcdir) and not a seperate build directory.
$(inst_libdir)/libpthread2.a: $(inst_libdir)/libpthread.a
	mv $< $@
	$(INSTALL_DATA) $(srcdir)/libpthread.a $<

$(inst_libdir)/libpthread2_pic.a: $(inst_libdir)/libpthread_pic.a
	mv $< $@
	$(INSTALL_DATA) $(srcdir)/libpthread_pic.a $<

$(inst_libdir)/libpthread_syms.a: $(srcdir)/libpthread_syms.a $(+force)
	$(do-install)

libc-link.so = $(common-objpfx)libc.so

extra-B-pthread.so = -B$(common-objpfx)htl/
LDFLAGS-pthread.so = -Wl,--enable-new-dtags,-z,nodelete,-z,initfirst

include ../Rules

# Make sure we link with the thread library.
ifeq ($(build-shared),yes)
$(addprefix $(objpfx), \
  $(filter-out $(tests-static) $(xtests-static) $(tests-reverse) \
    $(tests-nolibpthread), \
    $(tests) $(tests-internal) $(xtests) $(test-srcs) $(tests-container))): \
	$(objpfx)libpthread.so
# $(objpfx)../libc.so is used instead of $(common-objpfx)libc.so,
# since otherwise libpthread.so comes before libc.so when linking.
$(addprefix $(objpfx), $(tests-reverse)): \
  $(objpfx)../libc.so $(objpfx)libpthread.so
$(objpfx)../libc.so: $(common-objpfx)libc.so ;
$(addprefix $(objpfx),$(tests-static) $(xtests-static)): $(srcdir)/libpthread_syms.a $(objpfx)libpthread.a
else
$(addprefix $(objpfx),$(tests) $(test-srcs)): $(srcdir)/libpthread_syms.a $(objpfx)libpthread.a
endif
