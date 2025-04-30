# Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#	Sub-makefile for NPTL portion of the library.
#
subdir	:= nptl

include ../Makeconfig

headers := pthread.h semaphore.h bits/semaphore.h \
	   bits/struct_mutex.h bits/struct_rwlock.h

extra-libs := libpthread
extra-libs-others := $(extra-libs)

routines = \
  alloca_cutoff \
  cancellation \
  cleanup \
  cleanup_compat \
  cleanup_defer \
  cleanup_defer_compat \
  cleanup_routine \
  elision-conf \
  elision-lock \
  elision-timed \
  elision-trylock \
  elision-unlock \
  events \
  futex-internal \
  libc-cleanup \
  libc_multiple_threads \
  lowlevellock \
  nptl-stack \
  nptl_deallocate_tsd \
  nptl_free_tcb \
  nptl_nthreads \
  nptl_setxid \
  nptlfreeres \
  old_pthread_cond_broadcast \
  old_pthread_cond_destroy \
  old_pthread_cond_init \
  old_pthread_cond_signal \
  old_pthread_cond_timedwait \
  old_pthread_cond_wait \
  pthread_atfork \
  pthread_attr_copy \
  pthread_attr_destroy \
  pthread_attr_extension \
  pthread_attr_getaffinity \
  pthread_attr_getdetachstate \
  pthread_attr_getguardsize \
  pthread_attr_getinheritsched \
  pthread_attr_getschedparam \
  pthread_attr_getschedpolicy \
  pthread_attr_getscope \
  pthread_attr_getsigmask \
  pthread_attr_getstack \
  pthread_attr_getstackaddr \
  pthread_attr_getstacksize \
  pthread_attr_init \
  pthread_attr_setaffinity \
  pthread_attr_setdetachstate \
  pthread_attr_setguardsize \
  pthread_attr_setinheritsched \
  pthread_attr_setschedparam \
  pthread_attr_setschedpolicy \
  pthread_attr_setscope \
  pthread_attr_setsigmask \
  pthread_attr_setsigmask_internal \
  pthread_attr_setstack \
  pthread_attr_setstackaddr \
  pthread_attr_setstacksize \
  pthread_barrier_destroy \
  pthread_barrier_init \
  pthread_barrier_wait \
  pthread_barrierattr_destroy \
  pthread_barrierattr_getpshared \
  pthread_barrierattr_init \
  pthread_barrierattr_setpshared \
  pthread_cancel \
  pthread_cleanup_upto \
  pthread_clockjoin \
  pthread_cond_broadcast \
  pthread_cond_destroy \
  pthread_cond_init \
  pthread_cond_signal \
  pthread_cond_wait \
  pthread_condattr_destroy \
  pthread_condattr_getclock \
  pthread_condattr_getpshared \
  pthread_condattr_init \
  pthread_condattr_setclock \
  pthread_condattr_setpshared \
  pthread_create \
  pthread_detach \
  pthread_equal \
  pthread_exit \
  pthread_getaffinity \
  pthread_getattr_default_np \
  pthread_getattr_np \
  pthread_getconcurrency \
  pthread_getcpuclockid \
  pthread_getname \
  pthread_getschedparam \
  pthread_getspecific \
  pthread_join \
  pthread_join_common \
  pthread_key_create \
  pthread_key_delete \
  pthread_keys \
  pthread_kill \
  pthread_kill_other_threads \
  pthread_mutex_cond_lock \
  pthread_mutex_conf \
  pthread_mutex_consistent \
  pthread_mutex_destroy \
  pthread_mutex_getprioceiling \
  pthread_mutex_init \
  pthread_mutex_lock \
  pthread_mutex_setprioceiling \
  pthread_mutex_timedlock \
  pthread_mutex_trylock \
  pthread_mutex_unlock \
  pthread_mutexattr_destroy \
  pthread_mutexattr_getprioceiling \
  pthread_mutexattr_getprotocol \
  pthread_mutexattr_getpshared \
  pthread_mutexattr_getrobust \
  pthread_mutexattr_gettype \
  pthread_mutexattr_init \
  pthread_mutexattr_setprioceiling \
  pthread_mutexattr_setprotocol \
  pthread_mutexattr_setpshared \
  pthread_mutexattr_setrobust \
  pthread_mutexattr_settype \
  pthread_once \
  pthread_rwlock_clockrdlock \
  pthread_rwlock_clockwrlock \
  pthread_rwlock_destroy \
  pthread_rwlock_init \
  pthread_rwlock_rdlock \
  pthread_rwlock_timedrdlock \
  pthread_rwlock_timedwrlock \
  pthread_rwlock_tryrdlock \
  pthread_rwlock_trywrlock \
  pthread_rwlock_unlock \
  pthread_rwlock_wrlock \
  pthread_rwlockattr_destroy \
  pthread_rwlockattr_getkind_np \
  pthread_rwlockattr_getpshared \
  pthread_rwlockattr_init \
  pthread_rwlockattr_setkind_np \
  pthread_rwlockattr_setpshared \
  pthread_self \
  pthread_setaffinity \
  pthread_setattr_default_np \
  pthread_setcancelstate \
  pthread_setcanceltype \
  pthread_setconcurrency \
  pthread_setname \
  pthread_setschedparam \
  pthread_setschedprio \
  pthread_setspecific \
  pthread_sigmask \
  pthread_sigqueue \
  pthread_spin_destroy \
  pthread_spin_init \
  pthread_spin_lock \
  pthread_spin_trylock \
  pthread_spin_unlock \
  pthread_testcancel \
  pthread_timedjoin \
  pthread_tryjoin \
  pthread_yield \
  sem_clockwait \
  sem_close \
  sem_destroy \
  sem_getvalue \
  sem_init \
  sem_open \
  sem_post \
  sem_routines \
  sem_timedwait \
  sem_unlink \
  sem_wait \
  tpp \
  unwind \
  vars \

static-only-routines = pthread_atfork
libpthread-routines = libpthread-compat
libpthread-shared-only-routines = libpthread-compat

# Pretend that libpthread.so is a linker script, so that the symbolic
# link is not installed.
install-lib-ldscripts = libpthread.so
$(inst_libdir)/libpthread.so:

# Since cancellation handling is in large parts handled using exceptions
# we have to compile some files with exception handling enabled, some
# even with asynchronous unwind tables.

# nptl-init.c contains sigcancel_handler().
CFLAGS-nptl-init.c += -fexceptions -fasynchronous-unwind-tables
# The unwind code itself,
CFLAGS-unwind.c += -fexceptions
CFLAGS-unwind-forcedunwind.c += -fexceptions -fasynchronous-unwind-tables

# The following three functions must be async-cancel safe.
CFLAGS-pthread_cancel.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pthread_setcancelstate.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pthread_setcanceltype.c += -fexceptions -fasynchronous-unwind-tables

# These are internal functions which similar functionality as setcancelstate
# and setcanceltype.
CFLAGS-cancellation.c += -fasynchronous-unwind-tables

# Calling pthread_exit() must cause the registered cancel handlers to
# be executed.  Therefore exceptions have to be thrown through this
# function.
CFLAGS-pthread_exit.c += -fexceptions

# Among others, __pthread_unwind is forwarded.  This function must handle
# exceptions.
CFLAGS-forward.c += -fexceptions

# The following are cancellation points.  Some of the functions can
# block and therefore temporarily enable asynchronous cancellation.
# Those must be compiled asynchronous unwind tables.
CFLAGS-pthread_testcancel.c += -fexceptions
CFLAGS-pthread_join.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pthread_timedjoin.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pthread_clockjoin.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pthread_once.c += $(uses-callbacks) -fexceptions \
			-fasynchronous-unwind-tables
CFLAGS-pthread_cond_wait.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-pthread_kill.c = -fexceptions -fasynchronous-unwind-tables
CFLAGS-sem_wait.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-sem_timedwait.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-sem_clockwait.c = -fexceptions -fasynchronous-unwind-tables

CFLAGS-futex-internal.c += -fexceptions -fasynchronous-unwind-tables

# Workaround for clang assembler problem with ifunc relocs, b/119574415
ifeq ($(with-clang),yes)
CFLAGS-pt-vfork.c += -no-integrated-as
endif

LDLIBS-tst-once5 = -lstdc++
CFLAGS-tst-thread_local1.o = -std=gnu++11
LDLIBS-tst-thread_local1 = -lstdc++
CFLAGS-tst-thread-exit-clobber.o = -std=gnu++11
LDLIBS-tst-thread-exit-clobber = -lstdc++
CFLAGS-tst-minstack-throw.o = -std=gnu++11
LDLIBS-tst-minstack-throw = -lstdc++

tests = tst-attr2 tst-attr3 tst-default-attr \
	tst-mutex5a tst-mutex7a \
	tst-mutexpi1 tst-mutexpi2 tst-mutexpi3 tst-mutexpi4 \
	tst-mutexpi5 tst-mutexpi5a tst-mutexpi6 tst-mutexpi7 tst-mutexpi7a \
	tst-mutexpi9 tst-mutexpi10 \
	tst-cond22 tst-cond26 \
	tst-robustpi1 tst-robustpi2 tst-robustpi3 tst-robustpi4 tst-robustpi5 \
	tst-robustpi6 tst-robustpi7 tst-robustpi9 \
	tst-rwlock2 tst-rwlock2a tst-rwlock2b tst-rwlock3 \
	tst-rwlock6 tst-rwlock7 tst-rwlock8 \
	tst-rwlock9 tst-rwlock10 tst-rwlock11 \
	tst-rwlock15 tst-rwlock17 tst-rwlock18 \
	tst-once5 \
	tst-sem17 \
	tst-tsd3 tst-tsd4 \
	tst-cancel4_1 tst-cancel4_2 \
	tst-cancel7 tst-cancel17 tst-cancel24 \
	tst-signal3 \
	tst-exec4 tst-exec5 \
	tst-stack2 tst-stack3 tst-stack4 \
	tst-pthread-attr-affinity \
	tst-pthread-attr-affinity-fail \
	tst-dlsym1 \
	tst-context1 \
	tst-sched1 \
	tst-initializers1 $(addprefix tst-initializers1-,\
			    c89 gnu89 c99 gnu99 c11 gnu11) \
	tst-thread_local1 \
	tst-robust-fork \
	tst-thread-exit-clobber tst-minstack-cancel tst-minstack-exit \
	tst-minstack-throw \
	tst-rwlock-pwn \
	tst-thread-affinity-pthread \
	tst-thread-affinity-pthread2 \
	tst-thread-affinity-sched \
	tst-pthread-defaultattr-free \
	tst-pthread-attr-sigmask \
	tst-pthread-timedlock-lockloop \
	tst-pthread-gdb-attach tst-pthread-gdb-attach-static \
	tst-pthread_exit-nothreads \
	tst-pthread_exit-nothreads-static \
	tst-thread-setspecific

tests-nolibpthread = \
  tst-pthread_exit-nothreads \
  tst-pthread_exit-nothreads-static \

tests-container =  tst-pthread-getattr

tests-internal := tst-robustpi8 tst-rwlock19 tst-rwlock20 \
		  tst-sem11 tst-sem12 tst-sem13 \
		  tst-barrier5 tst-signal7 tst-mutex8 tst-mutex8-static \
		  tst-mutexpi8 tst-mutexpi8-static \
		  tst-setgetname \

xtests = tst-setuid1 tst-setuid1-static tst-setuid2 \
	tst-mutexpp1 tst-mutexpp6 tst-mutexpp10 tst-setgroups \
	tst-mutexpp5 tst-mutexpp9

tests-time64 := \
  tst-cancel4_2-time64

# This test can run into task limits because of a linux kernel bug
# and then cause the make process to fail too, see bug 24537.
xtests += tst-eintr1

test-srcs = tst-oddstacklimit

gen-as-const-headers = unwindbuf.sym \
		       pthread-pi-defines.sym

gen-py-const-headers := nptl_lock_constants.pysym
pretty-printers := nptl-printers.py
tests-printers := test-mutexattr-printers test-mutex-printers \
		  test-condattr-printers test-cond-printers \
		  test-rwlockattr-printers test-rwlock-printers

# We must specify both CFLAGS and CPPFLAGS to override any
# compiler options the user might have provided that conflict
# with what we need e.g. user specifies CPPFLAGS with -O2 and
# we need -O0.
CFLAGS-test-mutexattr-printers.c := $(CFLAGS-printers-tests)
CFLAGS-test-mutex-printers.c := $(CFLAGS-printers-tests)
CFLAGS-test-condattr-printers.c := $(CFLAGS-printers-tests)
CFLAGS-test-cond-printers.c := $(CFLAGS-printers-tests)
CFLAGS-test-rwlockattr-printers.c := $(CFLAGS-printers-tests)
CFLAGS-test-rwlock-printers.c := $(CFLAGS-printers-tests)
CPPFLAGS-test-mutexattr-printers.c := $(CFLAGS-printers-tests)
CPPFLAGS-test-mutex-printers.c := $(CFLAGS-printers-tests)
CPPFLAGS-test-condattr-printers.c := $(CFLAGS-printers-tests)
CPPFLAGS-test-cond-printers.c := $(CFLAGS-printers-tests)
CPPFLAGS-test-rwlockattr-printers.c := $(CFLAGS-printers-tests)
CPPFLAGS-test-rwlock-printers.c := $(CFLAGS-printers-tests)

# Reuse the CFLAGS setting for the GDB attaching test.  It needs
# debugging information.
CFLAGS-tst-pthread-gdb-attach.c := $(CFLAGS-printers-tests)
CPPFLAGS-tst-pthread-gdb-attach.c := $(CFLAGS-printers-tests)
ifeq ($(build-shared)$(build-hardcoded-path-in-tests),yesno)
CPPFLAGS-tst-pthread-gdb-attach.c += -DDO_ADD_SYMBOL_FILE=1
else
CPPFLAGS-tst-pthread-gdb-attach.c += -DDO_ADD_SYMBOL_FILE=0
endif
CFLAGS-tst-pthread-gdb-attach-static.c := $(CFLAGS-printers-tests)
CPPFLAGS-tst-pthread-gdb-attach-static.c := \
  $(CFLAGS-printers-tests) -DDO_ADD_SYMBOL_FILE=0
# As of version 9.2, GDB cannot attach properly to PIE programs that
# were launched with an explicit ld.so invocation.
tst-pthread-gdb-attach-no-pie = yes

tests += tst-cancelx7 tst-cancelx17

ifeq ($(build-shared),yes)
tests += tst-compat-forwarder tst-audit-threads
tests-internal += tst-tls3 tst-tls3-malloc tst-tls5 tst-stackguard1
ifeq ($(have-z-execstack),yes)
# Test depends on a nested function, can't compile without rewrite
ifneq ($(with-clang),yes)
tests += tst-execstack
endif
endif
endif

modules-names = tst-tls3mod \
		tst-tls5mod tst-tls5moda tst-tls5modb tst-tls5modc \
		tst-tls5modd tst-tls5mode tst-tls5modf tst-stack4mod \
		tst-execstack-mod \
		tst-compat-forwarder-mod tst-audit-threads-mod1 \
		tst-audit-threads-mod2
extra-test-objs += $(addsuffix .os,$(strip $(modules-names))) \
		   tst-cleanup4aux.o tst-cleanupx4aux.o
test-extras += tst-cleanup4aux tst-cleanupx4aux

# This test exercises compat symbols removed in glibc 2.34.
ifdef have-GLIBC_2.33
tests += tst-cleanup4
ifeq ($(build-shared),yes)
tests += tst-cleanupx4
endif
endif

tst-tls3mod.so-no-z-defs = yes
tst-tls5mod.so-no-z-defs = yes
tst-tls5moda.so-no-z-defs = yes
tst-tls5modb.so-no-z-defs = yes
tst-tls5modc.so-no-z-defs = yes
tst-tls5modd.so-no-z-defs = yes
tst-tls5mode.so-no-z-defs = yes
tst-tls5modf.so-no-z-defs = yes

ifeq ($(build-shared),yes)

# Set the `multidir' variable by grabbing the variable from the compiler.
# We do it once and save the result in a generated makefile.
# Suppress not-working clang attempt to include 32 and x32 multidirs.
-include $(objpfx)multidir.mk
$(objpfx)multidir.mk: $(common-objpfx)config.make
	$(make-target-directory)
	dir="."; \
	echo "multidir := $$dir" > $@T
	mv -f $@T $@

endif

CFLAGS-ftrylockfile.c += $(libio-mtsafe)
CFLAGS-funlockfile.c += $(libio-mtsafe)

link-libc-static := $(common-objpfx)libc.a $(static-gnulib) \
		    $(common-objpfx)libc.a

tests-static += tst-stackguard1-static \
		tst-cancel24-static \
		tst-mutex8-static tst-mutexpi8-static tst-sem11-static \
		tst-sem12-static tst-cond11-static \
		tst-pthread-gdb-attach-static \
		tst-pthread_exit-nothreads-static

tests += tst-cancel24-static

tests-internal += tst-sem11-static tst-sem12-static tst-stackguard1-static
xtests-static += tst-setuid1-static

ifeq ($(run-built-tests),yes)
tests-special += $(objpfx)tst-stack3-mem.out $(objpfx)tst-oddstacklimit.out
ifeq ($(build-shared),yes)
tests-special += $(objpfx)tst-tls6.out
endif
endif

ifeq (,$(CXX))
# These tests require a C++ compiler and runtime.
tests-unsupported += tst-cancel24 tst-cancel24-static tst-once5 \
  tst-thread-exit-clobber tst-minstack-throw
endif
# These tests require a C++ compiler and runtime with thread_local support.
ifneq ($(have-cxx-thread_local),yes)
tests-unsupported += tst-thread_local1
endif

include ../Rules

ifeq (yes,$(build-shared))
# Make sure these things are built in the `make lib' pass so they can be used
# to run programs during the `make others' pass.
lib-noranlib: $(addprefix $(objpfx),$(extra-objs))
endif


# 'pthread_self' is a simple memory or register load.  Setting up the
# stack frame is more work than the actual operation.  Disable the
# frame creation entirely.  This will help applications which call the
# function frequently to get a thread-specific handle.
CFLAGS-pthread_self.os += -fomit-frame-pointer

# Run the cancellation and cleanup tests also for the modern, exception-based
# implementation.  For this we have to pass the -fexceptions parameter.
CFLAGS-tst-cancelx7.c += -fexceptions
CFLAGS-tst-cancelx17.c += -fexceptions
CFLAGS-tst-cleanupx4.c += -fexceptions
CFLAGS-tst-cleanupx4aux.c += -fexceptions
CFLAGS-tst-initializers1.c += -W -Wall -Werror
CFLAGS-tst-initializers1-< = $(CFLAGS-tst-initializers1.c) \
			     $(patsubst tst-initializers1-%.c,-std=%,$<)
CFLAGS-tst-initializers1-c89.c += $(CFLAGS-tst-initializers1-<)
CFLAGS-tst-initializers1-c99.c += $(CFLAGS-tst-initializers1-<)
CFLAGS-tst-initializers1-c11.c += $(CFLAGS-tst-initializers1-<)
CFLAGS-tst-initializers1-gnu89.c += $(CFLAGS-tst-initializers1-<)
CFLAGS-tst-initializers1-gnu99.c += $(CFLAGS-tst-initializers1-<)
CFLAGS-tst-initializers1-gnu11.c += $(CFLAGS-tst-initializers1-<)

tst-cancel7-ARGS = --command "exec $(host-test-program-cmd)"
tst-cancelx7-ARGS = $(tst-cancel7-ARGS)

tst-stack3-ENV = MALLOC_TRACE=$(objpfx)tst-stack3.mtrace \
		 LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
$(objpfx)tst-stack3-mem.out: $(objpfx)tst-stack3.out
	$(common-objpfx)malloc/mtrace $(objpfx)tst-stack3.mtrace > $@; \
	$(evaluate-test)
generated += tst-stack3-mem.out tst-stack3.mtrace

tst-stack4mod.sos=$(shell for i in 0 1 2 3 4 5 6 7 8 9 10 \
				   11 12 13 14 15 16 17 18 19; do \
			    for j in 0 1 2 3 4 5 6 7 8 9 10 \
				     11 12 13 14 15 16 17 18 19; do \
			      echo $(objpfx)tst-stack4mod-$$i-$$j.so; \
			    done; done)
$(objpfx)tst-stack4.out: $(tst-stack4mod.sos)
$(tst-stack4mod.sos): $(objpfx)tst-stack4mod.so
	cp -f $< $@
clean:
	rm -f $(tst-stack4mod.sos)

$(objpfx)tst-cleanup4: $(objpfx)tst-cleanup4aux.o
$(objpfx)tst-cleanupx4: $(objpfx)tst-cleanupx4aux.o

LDFLAGS-tst-tls3 = -rdynamic
$(objpfx)tst-tls3.out: $(objpfx)tst-tls3mod.so

LDFLAGS-tst-tls3-malloc = -rdynamic
$(objpfx)tst-tls3-malloc.out: $(objpfx)tst-tls3mod.so

$(objpfx)tst-tls5: $(objpfx)tst-tls5mod.so
LDFLAGS-tst-tls5 = -Wl,--no-as-needed
LDFLAGS-tst-tls5mod.so = -Wl,-soname,tst-tls5mod.so

ifeq ($(build-shared),yes)
$(objpfx)tst-tls6.out: tst-tls6.sh $(objpfx)tst-tls5 \
		       $(objpfx)tst-tls5moda.so $(objpfx)tst-tls5modb.so \
		       $(objpfx)tst-tls5modc.so $(objpfx)tst-tls5modd.so \
		       $(objpfx)tst-tls5mode.so $(objpfx)tst-tls5modf.so
	$(BASH) $< $(common-objpfx) '$(test-via-rtld-prefix)' \
	  '$(test-wrapper-env)' '$(run-program-env)'; \
	$(evaluate-test)
endif

LDLIBS-tst-cancel24 = -Wl,--no-as-needed -lstdc++
LDLIBS-tst-cancel24-static = $(LDLIBS-tst-cancel24)

ifeq ($(build-shared),yes)

generated += multidir.mk tst-tls6.out
endif

tst-exec4-ARGS = $(host-test-program-cmd)

$(objpfx)tst-execstack.out: $(objpfx)tst-execstack-mod.so
LDFLAGS-tst-execstack = -Wl,-z,noexecstack
CFLAGS-tst-execstack-mod.c += -Wno-trampolines

tst-stackguard1-ARGS = --command "$(host-test-program-cmd) --child"
tst-stackguard1-static-ARGS = --command "$(objpfx)tst-stackguard1-static --child"

ifeq ($(run-built-tests),yes)
$(objpfx)tst-oddstacklimit.out: $(objpfx)tst-oddstacklimit $(objpfx)tst-basic1
	$(test-program-prefix) $< --command '$(host-test-program-cmd)' > $@; \
	$(evaluate-test)
endif

$(objpfx)tst-compat-forwarder: $(objpfx)tst-compat-forwarder-mod.so

tst-mutex10-ENV = GLIBC_TUNABLES=glibc.elision.enable=1

# Protect against a build using -Wl,-z,now.
LDFLAGS-tst-audit-threads-mod1.so = -Wl,-z,lazy
LDFLAGS-tst-audit-threads-mod2.so = -Wl,-z,lazy
LDFLAGS-tst-audit-threads = -Wl,-z,lazy
$(objpfx)tst-audit-threads: $(objpfx)tst-audit-threads-mod2.so
$(objpfx)tst-audit-threads.out: $(objpfx)tst-audit-threads-mod1.so
tst-audit-threads-ENV = LD_AUDIT=$(objpfx)tst-audit-threads-mod1.so

# The test uses dlopen indirectly and would otherwise load system
# objects.
tst-setuid1-static-ENV = \
  LD_LIBRARY_PATH=$(ld-library-path):$(common-objpfx)elf:$(common-objpfx)nss

# The tests here better do not run in parallel.
ifeq ($(run-built-tests),yes)
ifneq ($(filter %tests,$(MAKECMDGOALS)),)
.NOTPARALLEL:
endif
endif
