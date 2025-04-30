# Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#	Sub-makefile for real-time portion of the library.
#
subdir	:= rt

include ../Makeconfig

headers	:= aio.h mqueue.h bits/mqueue.h bits/mqueue2.h

routines = \
  shm_open \
  shm_unlink \

librt-routines = librt-compat
librt-shared-only-routines = librt-compat

$(librt-routines-var) += \
  aio_cancel \
  aio_error \
  aio_fsync \
  aio_misc \
  aio_notify \
  aio_read \
  aio_read64 \
  aio_return \
  aio_sigqueue \
  aio_suspend \
  aio_write \
  aio_write64 \
  lio_listio \
  lio_listio64 \
  mq_close \
  mq_getattr \
  mq_notify \
  mq_open \
  mq_receive \
  mq_send \
  mq_setattr \
  mq_timedreceive \
  mq_timedsend \
  mq_unlink \
  timer_create \
  timer_delete \
  timer_getoverr \
  timer_gettime \
  timer_settime \

ifeq ($(pthread-in-libc),yes)
# Pretend that librt.so is a linker script, so that the symbolic
# link is not installed.
install-lib-ldscripts = librt.so
$(inst_libdir)/librt.so:
endif

tests := tst-shm tst-timer tst-timer2 \
	 tst-aio tst-aio64 tst-aio2 tst-aio3 tst-aio4 tst-aio5 tst-aio6 \
	 tst-aio7 tst-aio8 tst-aio9 tst-aio10 \
	 tst-mqueue1 tst-mqueue2 tst-mqueue3 tst-mqueue4 \
	 tst-mqueue5 tst-mqueue6 tst-mqueue7 tst-mqueue8 tst-mqueue9 \
	 tst-timer3 tst-timer4 tst-timer5 \
	 tst-cpuclock2 tst-cputimer1 tst-cputimer2 tst-cputimer3 \
	 tst-shm-cancel \
	 tst-mqueue10
tests-internal := tst-timer-sigmask

tests-time64 := \
  tst-aio6-time64 \
  tst-cpuclock2-time64 \
  tst-mqueue1-time64 \
  tst-mqueue2-time64 \
  tst-mqueue4-time64 \
  tst-mqueue8-time64 \
  tst-mqueue10-time64 \
  tst-timer4-time64

extra-libs := librt
extra-libs-others := $(extra-libs)

include ../Rules

CFLAGS-aio_suspend.c += -fexceptions
CFLAGS-mq_timedreceive.c += -fexceptions -fasynchronous-unwind-tables
CFLAGS-mq_timedsend.c += -fexceptions -fasynchronous-unwind-tables

LDFLAGS-rt.so = -Wl,--enable-new-dtags,-z,nodelete

$(objpfx)librt.so: $(shared-thread-library)

ifneq ($(pthread-in-libc),yes)
ifeq (yes,$(build-shared))
$(addprefix $(objpfx),$(tests) $(tests-internal)): \
	$(objpfx)librt.so $(shared-thread-library)
else
$(addprefix $(objpfx),$(tests)) $(tests-internal): \
	$(objpfx)librt.a $(static-thread-library)
endif
endif # !$(pthread-in-libc)

tst-mqueue7-ARGS = -- $(host-test-program-cmd)
