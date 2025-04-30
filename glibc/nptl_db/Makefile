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

# Makefile for NPTL debug library subdirectory of GNU C Library.

subdir          := nptl_db

include ../Makeconfig

extra-libs = libthread_db
extra-libs-others := $(extra-libs)

headers         = proc_service.h thread_db.h sys/procfs.h

libthread_db-routines = td_init td_log td_ta_new td_ta_delete \
			td_ta_get_nthreads td_ta_get_ph \
			td_ta_map_id2thr td_ta_map_lwp2thr \
			td_ta_thr_iter td_ta_tsd_iter \
			td_thr_get_info td_thr_getfpregs td_thr_getgregs \
			td_thr_getxregs td_thr_getxregsize td_thr_setfpregs \
			td_thr_setgregs td_thr_setprio td_thr_setsigpending \
			td_thr_setxregs td_thr_sigsetmask td_thr_tsd \
			td_thr_validate td_thr_dbsuspend td_thr_dbresume \
			td_ta_setconcurrency td_ta_enable_stats \
			td_ta_reset_stats td_ta_get_stats td_ta_event_addr \
			td_thr_event_enable td_thr_set_event \
			td_thr_clear_event td_thr_event_getmsg \
			td_ta_set_event td_ta_event_getmsg \
			td_ta_clear_event td_symbol_list \
			td_thr_tlsbase td_thr_tls_get_addr \
			fetch-value

libthread_db-inhibit-o = $(filter-out .os,$(object-suffixes))

# The ps_* callback functions are not defined.
libthread_db.so-no-z-defs = yes

tests-special += $(objpfx)db-symbols.out

include ../Rules

$(objpfx)db-symbols.out: $(objpfx)db-symbols.v.i \
			 $(objpfx)libc.symlist-private
	LC_ALL=C $(AWK) -f $^ > $@; \
	$(evaluate-test)

$(objpfx)db-symbols.v.i: db-symbols.awk

$(objpfx)libc.symlist-private: $(..)scripts/abilist.awk \
  $(common-objpfx)libc.dynsym
	LC_ALL=C $(AWK) -v include_private=1 -f $^ > $@T
	mv -f $@T $@
