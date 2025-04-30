# sysdeps makefile fragment for all pthread-supporting configurations.
# Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

ifeq ($(subdir),rt)
$(objpfx)tst-timer: $(librt)
endif

ifneq (,$(filter $(subdir),htl nptl))
headers += threads.h

routines += thrd_current thrd_equal thrd_sleep thrd_yield pthread_atfork

$(libpthread-routines-var) += \
  call_once \
  cnd_broadcast \
  cnd_destroy \
  cnd_init \
  cnd_signal \
  cnd_timedwait \
  cnd_wait \
  mtx_destroy \
  mtx_init \
  mtx_lock \
  mtx_timedlock \
  mtx_trylock \
  mtx_unlock \
  pthread_atfork_compat \
  thrd_create \
  thrd_detach \
  thrd_exit \
  thrd_join \
  tss_create \
  tss_delete \
  tss_get \
  tss_set \

tests += tst-cnd-basic tst-mtx-trylock tst-cnd-broadcast \
	 tst-cnd-timedwait tst-thrd-detach tst-mtx-basic tst-thrd-sleep \
	 tst-mtx-recursive tst-tss-basic tst-call-once tst-mtx-timedlock \
	 \
	 tst-abstime \
	 tst-pt-align tst-pt-align3 \
	 tst-attr1 \
	 tst-backtrace1 \
	 tst-bad-schedattr \
	 tst-barrier1 tst-barrier2 tst-barrier3 tst-barrier4 \
	 tst-basic1 tst-basic2 tst-basic3 tst-basic4 tst-basic5 tst-basic6 \
	 tst-basic7 \
	 tst-cancel-self tst-cancel-self-cancelstate \
	 tst-cancel-self-canceltype tst-cancel-self-testcancel \
	 tst-cancel1 tst-cancel2 tst-cancel3 \
	 tst-cancel4 tst-cancel5 \
	 tst-cancel6 tst-cancel8 tst-cancel9 tst-cancel10 tst-cancel11 \
	 tst-cancel12 tst-cancel13 tst-cancel14 tst-cancel15 tst-cancel16 \
	 tst-cancel18 tst-cancel19 tst-cancel20 tst-cancel21 \
	 tst-cancel22 tst-cancel23 tst-cancel26 tst-cancel27 tst-cancel28 \
	 tst-cleanup0 tst-cleanup1 tst-cleanup2 tst-cleanup3 \
	 tst-clock1 \
	 tst-cond-except \
	 tst-cond1 tst-cond2 tst-cond3 tst-cond4 tst-cond5 tst-cond6 tst-cond7 \
	 tst-cond8 tst-cond9 tst-cond10 tst-cond11 tst-cond12 tst-cond13 \
	 tst-cond14 tst-cond15 tst-cond16 tst-cond17 tst-cond18 tst-cond19 \
	 tst-cond20 tst-cond21 tst-cond23 tst-cond24 tst-cond25 tst-cond27 \
	 tst-create-detached \
	 tst-detach1 \
	 tst-eintr2 tst-eintr3 tst-eintr4 tst-eintr5 \
	 tst-exec1 tst-exec2 tst-exec3 \
	 tst-exit1 tst-exit2 tst-exit3 \
	 tst-flock1 tst-flock2 \
	 tst-fork1 tst-fork2 tst-fork3 tst-fork4 \
	 tst-atfork1 \
	 tst-getpid3 \
	 tst-join1 tst-join2 tst-join3 tst-join4 tst-join5 tst-join6 tst-join7 \
	 tst-join8 tst-join9 tst-join10 tst-join11 tst-join12 tst-join13 \
	 tst-join14 tst-join15 \
	 tst-key1 tst-key2 tst-key3 tst-key4 \
	 tst-kill1 tst-kill2 tst-kill3 tst-kill4 tst-kill5 tst-kill6 \
	 tst-locale1 tst-locale2 \
	 tst-memstream \
	 tst-mutex-errorcheck tst-mutex1 tst-mutex2 tst-mutex3 tst-mutex4 \
	 tst-mutex5 tst-mutex6 tst-mutex7 tst-mutex7robust tst-mutex9 \
	 tst-mutex10 tst-mutex11 tst-pthread-mutexattr \
	 tst-once1 tst-once2 tst-once3 tst-once4 \
	 tst-pt-popen1 \
	 tst-raise1 \
	 tst-robust1 tst-robust2 tst-robust3 tst-robust4 tst-robust5 \
	 tst-robust6 tst-robust7 tst-robust9 tst-robust10 \
	 tst-rwlock1 tst-rwlock4 tst-rwlock5 tst-rwlock12 \
	 tst-rwlock13 tst-rwlock14 tst-rwlock16 \
	 tst-rwlock-tryrdlock-stall tst-rwlock-trywrlock-stall \
	 tst-sem1 tst-sem2 tst-sem3 tst-sem4 tst-sem5 tst-sem6 tst-sem7 \
	 tst-sem8 tst-sem9 tst-sem10 tst-sem14 tst-sem15 tst-sem16 \
	 tst-setuid3 \
	 tst-signal1 tst-signal2 \
	 tst-signal4 tst-signal5 tst-signal6 tst-signal8 \
	 tst-spin1 tst-spin2 tst-spin3 tst-spin4 \
	 tst-stack1 \
	 tst-stdio1 tst-stdio2 \
	 tst-pt-sysconf \
	 tst-pt-tls1 tst-pt-tls2 \
	 tst-tsd1 tst-tsd2 tst-tsd5 tst-tsd6 \
	 tst-umask1 \
	 tst-unload \
	 tst-unwind-thread \
	 tst-pt-vfork1 tst-pt-vfork2 tst-vfork1x tst-vfork2x \

tests-time64 := \
  tst-abstime-time64 \
  tst-cnd-timedwait-time64 \
  tst-cond11-time64 \
  tst-join14-time64 \
  tst-mtx-timedlock-time64 \
  tst-rwlock14-time64 \
  tst-sem5-time64 \
  tst-thrd-sleep-time64 \

static-only-routines = pthread_atfork

# Files which must not be linked with libpthread.
tests-nolibpthread += tst-unload

# GCC-4.9 compiles 'sprintf(NULL, ...)' into UD2 on x86_64 without -fno-builtin
CFLAGS-tst-cleanup2.c += -fno-builtin
CFLAGS-tst-cleanupx2.c += -fno-builtin

tests += tst-cancelx2 tst-cancelx3 tst-cancelx6 tst-cancelx8 tst-cancelx9 \
	 tst-cancelx4 tst-cancelx5 \
	 tst-cancelx10 tst-cancelx11 tst-cancelx12 tst-cancelx13 tst-cancelx14 \
	 tst-cancelx15 tst-cancelx16 tst-cancelx18 tst-cancelx20 tst-cancelx21 \
	 tst-cleanupx0 tst-cleanupx1 tst-cleanupx2 tst-cleanupx3

ifeq ($(build-shared),yes)
tests += tst-atfork2 tst-pt-tls4 tst-_res1 tst-fini1
tests-nolibpthread += tst-fini1
endif

modules-names += tst-atfork2mod tst-tls4moda tst-tls4modb \
		 tst-_res1mod1 tst-_res1mod2 tst-fini1mod
test-modules = $(addprefix $(objpfx),$(addsuffix .so,$(modules-names)))

tst-atfork2mod.so-no-z-defs = yes

ifeq ($(build-shared),yes)
# Build all the modules even when not actually running test programs.
tests: $(test-modules)
endif


tests-static += tst-locale1 tst-locale2 tst-cancel21-static

tests += tst-cancel21-static tst-cond11-static

# These tests are linked with libc before libpthread
tests-reverse += tst-cancel5 tst-cancel23 tst-vfork1x tst-vfork2x

ifeq ($(run-built-tests),yes)
ifeq ($(build-shared),yes)
tests-special += $(objpfx)tst-cleanup0-cmp.out
endif
endif

# The sigmask macro is deprecated and triggers are compiler warning
# which cannot be suppressed (as of GCC 9).
CFLAGS-tst-cancel4.c += -Wno-error
CFLAGS-tst-cancel5.c += -Wno-error
CFLAGS-tst-cancelx4.c += -Wno-error
CFLAGS-tst-cancelx5.c += -Wno-error

# Run the cancellation and cleanup tests also for the modern, exception-based
# implementation.  For this we have to pass the -fexceptions parameter.
CFLAGS-tst-cancelx2.c += -fexceptions
CFLAGS-tst-cancelx3.c += -fexceptions
CFLAGS-tst-cancelx4.c += -fexceptions
CFLAGS-tst-cancelx5.c += -fexceptions
CFLAGS-tst-cancelx6.c += -fexceptions
CFLAGS-tst-cancelx8.c += -fexceptions
CFLAGS-tst-cancelx9.c += -fexceptions
CFLAGS-tst-cancelx10.c += -fexceptions
CFLAGS-tst-cancelx11.c += -fexceptions
CFLAGS-tst-cancelx12.c += -fexceptions
CFLAGS-tst-cancelx13.c += -fexceptions
CFLAGS-tst-cancelx14.c += -fexceptions
CFLAGS-tst-cancelx15.c += -fexceptions
CFLAGS-tst-cancelx16.c += -fexceptions
CFLAGS-tst-cancelx18.c += -fexceptions
CFLAGS-tst-cancelx20.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-tst-cancelx21.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-tst-cleanupx0.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-tst-cleanupx1.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-tst-cleanupx2.c += -fexceptions
CFLAGS-tst-cleanupx3.c += -fexceptions
CFLAGS-tst-pt-align.c += $(stack-align-test-flags)
CFLAGS-tst-pt-align3.c += $(stack-align-test-flags)

tst-umask1-ARGS = $(objpfx)tst-umask1.temp

$(objpfx)tst-atfork2: $(shared-thread-library)
LDFLAGS-tst-atfork2 = -rdynamic
tst-atfork2-ENV = MALLOC_TRACE=$(objpfx)tst-atfork2.mtrace \
		  LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
$(objpfx)tst-atfork2mod.so: $(shared-thread-library)

ifeq ($(build-shared),yes)
$(objpfx)tst-atfork2.out: $(objpfx)tst-atfork2mod.so
endif

ifeq ($(build-shared),yes)
$(objpfx)tst-cleanup0.out: /dev/null $(objpfx)tst-cleanup0
	$(make-test-out) > $@ 2>&1; \
	$(evaluate-test)

$(objpfx)tst-cleanup0-cmp.out: $(..)sysdeps/pthread/tst-cleanup0.expect $(objpfx)tst-cleanup0.out
	cmp $^ > $@; \
	$(evaluate-test)
endif

$(objpfx)tst-cancel28: $(librt)

$(objpfx)tst-fini1: $(shared-thread-library) $(objpfx)tst-fini1mod.so

$(objpfx)tst-fini1mod.so: $(shared-thread-library)

$(objpfx)tst-_res1mod2.so: $(objpfx)tst-_res1mod1.so
LDFLAGS-tst-_res1mod1.so = -Wl,-soname,tst-_res1mod1.so
LDFLAGS-tst-_res1mod2.so = -Wl,-soname,tst-_res1mod2.so
$(objpfx)tst-_res1: $(objpfx)tst-_res1mod1.so $(objpfx)tst-_res1mod2.so \
		    $(shared-thread-library)

$(objpfx)tst-pt-tls4: $(shared-thread-library)
$(objpfx)tst-pt-tls4.out: $(objpfx)tst-tls4moda.so $(objpfx)tst-tls4modb.so

generated += tst-atfork2.mtrace

generated += $(objpfx)tst-atfork2.mtrace \
	     $(addsuffix .so,$(strip $(modules-names)))

tests-internal += tst-cancel25 tst-robust8

tests += tst-oncex3 tst-oncex4 tst-oncey3 tst-oncey4

modules-names += tst-join7mod

ifeq ($(build-shared),yes)
tests-static += tst-cond8-static
tests += tst-cond8-static
endif

CFLAGS-tst-oncex3.c += -fexceptions
CFLAGS-tst-oncex4.c += -fexceptions
CFLAGS-tst-oncey3.c += -fno-exceptions -fno-asynchronous-unwind-tables
CFLAGS-tst-oncey4.c += -fno-exceptions -fno-asynchronous-unwind-tables

$(objpfx)tst-join7: $(shared-thread-library)
$(objpfx)tst-join7.out: $(objpfx)tst-join7mod.so
$(objpfx)tst-join7mod.so: $(shared-thread-library)
LDFLAGS-tst-join7mod.so = -Wl,-soname,tst-join7mod.so

CFLAGS-tst-unwind-thread.c += -funwind-tables

endif
