# Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
#	Sub-makefile for resolv portion of the library.
#
subdir	:= resolv

include ../Makeconfig

headers	:= resolv.h bits/types/res_state.h \
	   netdb.h bits/netdb.h \
	   arpa/nameser.h arpa/nameser_compat.h \
	   sys/bitypes.h

routines := \
  dn_comp \
  dn_expand \
  dn_skipname \
  dns-canon \
  dns-host \
  dns-network \
  herror \
  inet_addr \
  inet_ntop \
  inet_pton \
  ns_makecanon \
  ns_name_compress \
  ns_name_ntop \
  ns_name_pack \
  ns_name_pton \
  ns_name_skip \
  ns_name_uncompress \
  ns_name_unpack \
  ns_samename \
  nsap_addr \
  nss_dns_functions \
  res-close \
  res-name-checking \
  res-state \
  res_context_hostalias \
  res_enable_icmp \
  res_get_nsaddr \
  res_hconf \
  res_init \
  res_libc \
  res_mkquery \
  res_nameinquery \
  res_queriesmatch \
  res_query \
  res_randomid \
  res_send \
  resolv_conf \
  resolv_context \
  # routines

tests = tst-aton tst-leaks tst-inet_ntop
tests-container = tst-leaks2

tests-internal += tst-inet_aton_exact


generate := mtrace-tst-leaks.out tst-leaks.mtrace tst-leaks2.mtrace

extra-libs := libresolv libnss_dns
ifeq ($(have-thread-library),yes)
extra-libs += libanl
routines += gai_sigqueue

tests += \
  tst-bug18665 \
  tst-bug18665-tcp \
  tst-ns_name \
  tst-ns_name_compress \
  tst-ns_name_pton \
  tst-res_hconf_reorder \
  tst-res_hnok \
  tst-resolv-basic \
  tst-resolv-binary \
  tst-resolv-edns \
  tst-resolv-network \
  tst-resolv-nondecimal \
  tst-resolv-res_init-multi \
  tst-resolv-search \
  tst-resolv-trailing \

# This test calls __res_context_send directly, which is not exported
# from libresolv.
tests-internal += tst-resolv-txnid-collision
tests-static += tst-resolv-txnid-collision

# These tests need libdl.
ifeq (yes,$(build-shared))
tests += \
  tst-resolv-ai_idn \
  tst-resolv-ai_idn-latin1 \
  tst-resolv-ai_idn-nolibidn2 \
  tst-resolv-canonname \
  tst-resolv-trustad \

# Needs resolv_context.
tests-internal += \
  tst-resolv-res_init \
  tst-resolv-res_init-thread \
  tst-resolv-res_ninit \
  tst-resolv-threads \

# Used by tst-resolv-ai_idn-nolibidn2 to disable libidn2 (by not
# providing any functions in libidn2.so.0).
modules-names += tst-no-libidn2
extra-test-objs += tst-no-libidn2.os
LDFLAGS-tst-no-libidn2.so = -Wl,-soname,libidn2.so.0

endif # $(build-shared)

# This test accesses __inet_ntop_length, an internal libc function.
tests-internal += tst-inet_pton

# This test accesses the __p_secstodate compat symbol.
ifeq ($(have-GLIBC_2.26)$(build-shared),yesyes)
tests += tst-p_secstodate
endif

# This test sends millions of packets and is rather slow.
xtests += tst-resolv-qtypes

# This test has dropped packet tests and runs for a long time.
xtests += tst-resolv-rotate
endif
extra-libs-others = $(extra-libs)
libresolv-routines := \
  base64 \
  compat-gethnamaddr \
  compat-hooks \
  inet_net_ntop \
  inet_net_pton \
  inet_neta \
  ns_date \
  ns_name \
  ns_netint \
  ns_parse \
  ns_print \
  ns_samedomain \
  ns_ttl \
  res-putget \
  res_data \
  res_debug \
  res_hostalias \
  res_isourserver \
  resolv-deprecated \
  # libresolv-routines

$(libanl-routines-var) += \
  gai_cancel \
  gai_error \
  gai_misc \
  gai_notify \
  gai_suspend \
  getaddrinfo_a \
  # $(libanl-routines-var)

libanl-routines += libanl-compat
libanl-shared-only-routines += libanl-compat

subdir-dirs = nss_dns
vpath %.c nss_dns

# Build only an empty shared libnss_dns.
libnss_dns-inhibit-o = $(filter-out .os,$(object-suffixes))

# Pretend that libnss_dns.so is a linker script, so that the symbolic link
# is not installed.
install-lib-ldscripts = libnss_dns.so
$(inst_libdir)/libnss_dns.so:

ifeq ($(run-built-tests),yes)
ifneq (no,$(PERL))
tests-special += $(objpfx)mtrace-tst-leaks.out $(objpfx)mtrace-tst-leaks2.out \
		 $(objpfx)mtrace-tst-resolv-res_ninit.out
endif
endif

generated += mtrace-tst-leaks.out tst-leaks.mtrace \
	     mtrace-tst-leaks2.out tst-leaks2.mtrace \
	     mtrace-tst-resolv-res_ninit.out tst-resolv-res_ninit.mtrace \

include ../Rules

LOCALES := en_US.UTF-8 en_US.ISO-8859-1
include ../gen-locales.mk

CFLAGS-res_hconf.c += -fexceptions

# The DNS NSS modules needs the resolver.
$(objpfx)libnss_dns.so: $(objpfx)libresolv.so

# The asynchronous name lookup code needs the thread library.
$(objpfx)libanl.so: $(shared-thread-library)

$(objpfx)tst-res_hconf_reorder: $(shared-thread-library)
tst-res_hconf_reorder-ENV = RESOLV_REORDER=on

$(objpfx)tst-leaks: $(objpfx)libresolv.so
tst-leaks-ENV = MALLOC_TRACE=$(objpfx)tst-leaks.mtrace \
		LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
$(objpfx)mtrace-tst-leaks.out: $(objpfx)tst-leaks.out
	$(common-objpfx)malloc/mtrace $(objpfx)tst-leaks.mtrace > $@; \
	$(evaluate-test)

tst-leaks2-ENV = MALLOC_TRACE=$(objpfx)tst-leaks2.mtrace \
		 LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
$(objpfx)mtrace-tst-leaks2.out: $(objpfx)tst-leaks2.out
	{ test -r $(objpfx)tst-leaks2.mtrace \
	|| ( echo "tst-leaks2.mtrace does not exist"; exit 77; ) \
	&& $(common-objpfx)malloc/mtrace $(objpfx)tst-leaks2.mtrace; } > $@; \
	$(evaluate-test)

tst-resolv-res_ninit-ENV = MALLOC_TRACE=$(objpfx)tst-resolv-res_ninit.mtrace \
			LD_PRELOAD=$(common-objpfx)/malloc/libc_malloc_debug.so
$(objpfx)mtrace-tst-resolv-res_ninit.out: $(objpfx)tst-resolv-res_ninit.out
	$(common-objpfx)malloc/mtrace \
	  $(objpfx)tst-resolv-res_ninit.mtrace > $@; \
	  $(evaluate-test)

$(objpfx)tst-bug18665-tcp: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-bug18665: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-ai_idn: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-ai_idn-latin1: \
  $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-ai_idn-nolibidn2: \
  $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-ai_idn.out: $(gen-locales)
$(objpfx)tst-resolv-ai_idn-latin1.out: $(gen-locales)
$(objpfx)tst-resolv-ai_idn-nolibidn2.out: \
  $(gen-locales) $(objpfx)tst-no-libidn2.so
$(objpfx)tst-resolv-basic: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-binary: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-edns: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-network: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-res_init: $(objpfx)libresolv.so
$(objpfx)tst-resolv-res_init-multi: $(objpfx)libresolv.so \
  $(shared-thread-library)
$(objpfx)tst-resolv-res_init-thread: $(objpfx)libresolv.so \
  $(shared-thread-library)
$(objpfx)tst-resolv-nondecimal: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-qtypes: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-rotate: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-search: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-trailing: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-threads: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-txnid-collision: $(objpfx)libresolv.a \
  $(static-thread-library)
$(objpfx)tst-resolv-canonname: $(objpfx)libresolv.so $(shared-thread-library)
$(objpfx)tst-resolv-trustad: $(objpfx)libresolv.so $(shared-thread-library)

$(objpfx)tst-ns_name: $(objpfx)libresolv.so
$(objpfx)tst-ns_name.out: tst-ns_name.data
$(objpfx)tst-ns_name_compress: $(objpfx)libresolv.so
$(objpfx)tst-ns_name_pton: $(objpfx)libresolv.so
$(objpfx)tst-res_hnok: $(objpfx)libresolv.so
$(objpfx)tst-p_secstodate: $(objpfx)libresolv.so
