# Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

subdir := hurd

include ../Makeconfig

headers = hurd.h $(interface-headers) \
	  $(addprefix hurd/,fd.h id.h port.h signal.h sigpreempt.h ioctl.h\
			    userlink.h resource.h lookup.h)

inline-headers = hurd.h $(addprefix hurd/,fd.h signal.h \
					  userlink.h port.h)

# The RPC interfaces go in a separate library.
interface-library := libhurduser
user-interfaces		:= $(addprefix hurd/,\
				       auth startup \
				       process process_request \
				       msg msg_reply msg_request \
				       exec exec_startup crash interrupt \
				       fs fsys io io_reply io_request \
				       term tioctl socket ifsock \
				       login password pfinet pci \
				       )
server-interfaces	:= hurd/msg faultexc

routines = hurdstartup hurdinit \
	   hurdid hurdpid hurdrlimit hurdprio hurdexec hurdselect \
	   hurdlookup lookup-retry lookup-at \
	   get-host set-host \
	   path-lookup \
	   setauth \
	   pid2task task2pid \
	   geteuids seteuids getumask fchroot \
	   hurdsock hurdauth \
	   hurdchdir hurdfchdir \
	   privports \
	   msgportdemux \
	   fopenport \
	   vpprintf \
	   ports-get ports-set hurdports hurdmsg \
	   errno-loc \
	   hurdlock \
	   $(sig) $(dtable) $(inlines) \
	   fd-cleanup port-cleanup report-wait xattr
sig	= hurdsig hurdfault siginfo hurd-raise preempt-sig \
	  trampoline longjmp-ts catch-exc exc2signal hurdkill sigunwind \
	  thread-self thread-cancel intr-msg catch-signal
dtable	= dtable port2fd new-fd alloc-fd intern-fd \
	  getdport openport \
	  fd-close fd-read fd-write hurdioctl ctty-input ctty-output
inlines = $(inline-headers:%.h=%-inlines)

# XXX this is a temporary hack; see hurdmalloc.h
routines += hurdmalloc

# Binary compatibility for libc.so.0.2[GLIBC_2.0].
ifeq ($(build-shared),yes)
routines += compat-20
endif

shared-only-routines = compat-20

# For each of the $(inline-headers), generate a trivial source
# file that will #include it to define its inline functions as real functions.
$(inlines:%=$(objpfx)%.c): $(objpfx)%-inlines.c: %.h
	(h="`echo $(subst /,_,$*) | tr '[a-z]' '[A-Z]'`"; \
	 echo "#define _$${h}_H_EXTERN_INLINE /* Define real function.  */"; \
	 echo "#define _$${h}_H_HIDDEN_DEF    /* Declare hidden definition.  */"; \
	 echo '#include "$<"') > $@-new
	mv -f $@-new $@
generated += $(inlines:=.c)

# Avoid ssp before TLS is initialized.
CFLAGS-hurdstartup.o = $(no-stack-protector)
CFLAGS-RPC_exec_startup_get_info.o = $(no-stack-protector)

# Make generated headers compatible with all support standards
migheaderpipe := | sed -e 's/\<ino64_t\>/__ino64_t/' -e 's/\<loff_t\>/__loff_t/'
include ../mach/Machrules
libhurduser-routines += RPC_gsync_wait_intr
include ../Rules

# intr-rpc.defs defines the INTR_INTERFACE macro to make the generated RPC
# stubs import <hurd/signal.h> and #define __mach_msg to
# _hurd_intr_rpc_mach_msg.
user-MIGFLAGS += -imacros intr-rpc.defs

# The special exc server for sigthread faults uses a special prefix.
MIGFLAGS-faultexc = -prefix _hurdsig_fault_

# We need this static dependency to get faultexc.h generated the first time.
$(objpfx)hurdfault.o $(objpfx)hurdfault.d: \
	$(objpfx)faultexc_server.h $(objpfx)faultexc_server.c
