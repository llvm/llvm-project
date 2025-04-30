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
#	Sub-makefile for sunrpc portion of the library.
#
subdir	:= sunrpc

include ../Makeconfig

ifdef have-GLIBC_2.31

# The code in this subdirectory is taken from Sun's RPCSRC-4.0
# distribution with some additional changes from the TI-RPC package
# which is also available from Sun.  The files are heavily changed to
# compile cleanly and to fit in the GNU environment.  All the code
# from Sun's rpc and etc subdirectories is in this directory;
# the rpc subdirectory contains only the header files.  Other than
# that, several files were renamed so as not to exceed 14-character
# file name limits:
#
#	authunix_prot.c -> authuxprot.c
#	clnt_generic.c -> clnt_gen.c
#	clnt_perror.c -> clnt_perr.c
#	clnt_simple.c -> clnt_simp.c
#	get_myaddress.c -> get_myaddr.c
#	pmap_getmaps.c -> pm_getmaps.c
#	pmap_getport.c -> pm_getport.c
#	rpc_callmsg.c -> rpc_cmsg.c
#	rpc_commondata.c -> rpc_common.c
#	rpc_dtablesize.c -> rpc_dtable.c
#	svc_auth_unix.c -> svc_authux.c
#	xdr_reference.c -> xdr_ref.c

ifeq ($(build-shared),yes)
need-export-routines := auth_des auth_unix clnt_gen clnt_perr clnt_tcp \
			clnt_udp get_myaddr key_call netname pm_getport \
			rpc_thread svc svc_tcp svc_udp xcrypt xdr_array xdr \
			xdr_intXX_t xdr_mem xdr_ref xdr_sizeof xdr_stdio \
			svc_run

routines := auth_none authuxprot clnt_raw clnt_simp \
	    rpc_dtable getrpcport pmap_clnt pm_getmaps pmap_prot pmap_prot2 \
	    pmap_rmt rpc_prot rpc_common rpc_cmsg svc_auth svc_authux svc_raw \
	    svc_simple xdr_float xdr_rec publickey authdes_prot \
	    des_crypt des_impl des_soft key_prot openchild rtime svcauth_des \
	    clnt_unix svc_unix create_xid $(need-export-routines) \
	    rpc_gethostbyname
# We only add the RPC for compatibility to libc.so.
shared-only-routines = $(routines)
endif

tests = tst-xdrmem tst-xdrmem2 test-rpcent tst-udp-error tst-udp-timeout \
  tst-udp-nonblocking
xtests := tst-getmyaddr

ifeq ($(have-thread-library),yes)
xtests += thrsvc
tests += tst-udp-garbage tst-svc_register
endif

ifeq (yes,$(build-shared))
rpc-compat-routines = $(addprefix compat-,$(need-export-routines))
rpc-compat-routines.os = $(addprefix $(objpfx), \
				     $(addsuffix .os,$(rpc-compat-routines)))
extra-objs += $(addsuffix .os,$(rpc-compat-routines))
endif

include ../Rules

ifeq (yes,$(build-shared))
subdir_lib: $(objpfx)librpc_compat_pic.a
$(objpfx)librpc_compat_pic.a: $(rpc-compat-routines.os)
	$(AR) cr$(verbose) $@ $^
$(rpc-compat-routines.os): $(objpfx)compat-%.os: %.c $(before-compile)
	$(compile-command.c) -DEXPORT_RPC_SYMBOLS
endif

CFLAGS-auth_unix.c += -fexceptions
CFLAGS-key_call.c += -fexceptions
CFLAGS-pmap_rmt.c += -fexceptions
CFLAGS-clnt_perr.c += -fexceptions
CFLAGS-openchild.c += -fexceptions

$(objpfx)tst-getmyaddr: $(common-objpfx)linkobj/libc.so
$(objpfx)tst-xdrmem: $(common-objpfx)linkobj/libc.so
$(objpfx)tst-xdrmem2: $(common-objpfx)linkobj/libc.so
$(objpfx)tst-udp-error: $(common-objpfx)linkobj/libc.so
$(objpfx)tst-svc_register: \
  $(common-objpfx)linkobj/libc.so $(shared-thread-library)

$(objpfx)thrsvc: $(common-objpfx)linkobj/libc.so $(shared-thread-library)

$(objpfx)tst-udp-timeout: $(common-objpfx)linkobj/libc.so
$(objpfx)tst-udp-nonblocking: $(common-objpfx)linkobj/libc.so
$(objpfx)tst-udp-garbage: \
  $(common-objpfx)linkobj/libc.so $(shared-thread-library)

else # !have-GLIBC_2.31

routines = $(routines-for-nss)

include ../Rules

ifeq (yes,$(build-shared))
# Create an empty archive, so that Makerules can reference it.
subdir_lib: $(objpfx)librpc_compat_pic.a
$(objpfx)librpc_compat_pic.a: Makefile
	$(make-target-directory)
	$(AR) cr $@
endif


endif # !have-GLIBC_2.31
