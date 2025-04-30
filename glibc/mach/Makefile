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

subdir := mach

include ../Makeconfig

headers = mach_init.h mach.h mach_error.h mach-shortcuts.h mach/mach_traps.h \
	  $(interface-headers) mach/mach.h mach/mig_support.h mach/error.h \
	  $(lock-headers) machine-sp.h bits/mach/param.h
lock = spin-solid spin-lock mutex-init mutex-solid
lock-headers = lock-intern.h spin-lock.h
routines = $(mach-syscalls) $(mach-shortcuts) \
	   mach_init mig_strncpy mig_memcpy msg \
	   mig-alloc mig-dealloc mig-reply \
	   msg-destroy msgserver \
	   mach_error errstring error_compat errsystems \
	   devstream setup-thread $(lock)

# The RPC interfaces go in a separate library.
interface-library := libmachuser

# We produce mach_interface.h from mach.defs because there is already a
# <mach/mach.h> that is not the interface header (thanks CMU).
mach-interface-list := $(patsubst mach,mach_interface,$(mach-interface-list))
mach/mach_interface.defs = mach/mach.defs

# Similarly for memory_object_user vs. memory_object
mach-interface-list := $(patsubst memory_object,memory_object_user,\
				  $(mach-interface-list))
mach/memory_object_user.defs = mach/memory_object.defs

user-interfaces := $(addprefix mach/,$(mach-interface-list)	\
			       	     mach_port mach_host exc	\
		    )\
		   $(addprefix device/,device device_request)

server-interfaces := mach/exc

# Clear any environment	value.
generated =

# Avoid ssp before TLS is initialized.
CFLAGS-mach_init.o = $(no-stack-protector)
CFLAGS-RPC_vm_statistics.o = $(no-stack-protector)
CFLAGS-RPC_vm_map.o = $(no-stack-protector)
CFLAGS-RPC_vm_protect.o = $(no-stack-protector)
CFLAGS-RPC_i386_set_gdt.o = $(no-stack-protector)
CFLAGS-RPC_i386_set_ldt.o = $(no-stack-protector)
CFLAGS-RPC_task_get_special_port.o = $(no-stack-protector)

# Translate GNU names for CPUs into the names used in Mach header files.
mach-machine = $(patsubst powerpc,ppc,$(base-machine))

# Define mach-syscalls and sysno-*.
ifndef inhibit_mach_syscalls
-include $(objpfx)mach-syscalls.mk
endif
$(objpfx)mach-syscalls.mk: syscalls.awk Makefile libc-modules.h
# Go kludges!!!
	$(make-target-directory)
# We must use $(CFLAGS) to get -O flags that affect #if's in header files.
	echo '#include <mach/syscall_sw.h>' | \
	$(CC) $(CFLAGS) $(CPPFLAGS) -E -MD -MP -MF $@-dep -MT $@ \
	      -x c-header - \
	      -D_MACH_`echo $(mach-machine) | tr a-z A-Z`_SYSCALL_SW_H_=1 | \
	sed -n -e 's/^kernel_trap(\(.*\),\([-0-9]*\),\([0-9]*\))$$/\1 \2 \3/p'\
	| $(AWK) -f $< > $@-new
ifneq (,$(objpfx))
	sed $(sed-remove-objpfx) $@-dep >> $@-new
else
	cat $@-dep >> $@-new
endif
	rm -f $@-dep
	mv -f $@-new $@
generated += mach-syscalls.mk

ifndef mach-syscalls
# We cannot make the deps properly until we know how to make the system
# call functions, and for that we must know what they all are.
no_deps=t
else
$(mach-syscalls:%=$(objpfx)%.S): $(objpfx)%.S: $(objpfx)mach-syscalls.mk
	(echo '#include <sysdep.h>'; \
	 echo 'kernel_trap (__$*,$(sysno-$*),$(nargs-$*))'; \
	 echo 'weak_alias (__$*, $*)'; \
	 echo 'libc_hidden_def (__$*)') > $@-new
	 mv -f $@-new $@
generated += $(mach-syscalls:=.S)
endif	# mach-syscalls

# These syscalls that look like RPCs actually have no RPC equivalents.
mach-shortcuts := $(filter-out device_writev_request \
			       device_read_overwrite_request \
			       device_read_overwrite \
			       vm_read_overwrite \
			       thread_switch \
			       ,\
		  $(patsubst syscall_%,%,$(filter syscall_%,$(mach-syscalls))))

ifndef mach-shortcuts
# Forget about mach_interface.defs for this run.  On the next run,
# $(mach-shortcuts) will be set, and that will change how
# mach_interface.defs is processed: it will get the -D flags below.
user-interfaces := $(filter-out $(mach-interface-list:%=mach/%) \
				mach/mach_port mach/mach_host mach/mach4 mach/gnumach \
				device/device_request,\
				$(user-interfaces))
endif

# For some reason, the calls in this file are `protected' by an ifdef.
MIGFLAGS-mach/mach4=-DMACH_PCSAMPLE

# Make the MiG stubs for $(mach-shortcuts) be CALL_rpc.
migdefines += $(foreach call,$(mach-shortcuts),-D$(call)=$(call)_rpc)
$(mach-interface-list:%=mach/%.uh) \
  mach/mach_host.uh mach/mach_port.uh: $(objpfx)mach-syscalls.mk

ifdef mach-shortcuts
$(mach-shortcuts:%=$(objpfx)%.c): $(objpfx)%.c: shortcut.awk \
						$(objpfx)RPC_%_rpc.c
	$(AWK)	-v alias=$* -v call=__$* -v rpc=__$*_rpc \
		-v syscall=__syscall_$* -f $^ > $@-new
	mv -f $@-new $@
generated += $(mach-shortcuts:%=%.c)
endif	# mach-shortcuts

# Generate mach-shortcuts.h, which will contain the prototypes for the
# shortcutted kernel RPC functions.
$(objpfx)mach-shortcuts.h: $(mach-interface-list:%=$(objpfx)mach/%.h) \
			   $(objpfx)mach/mach_host.h \
			   $(objpfx)mach/mach_port.h
# The first line gets us one paragraph per line, with @s separating real lines.
# The second line selects paragraphs for the shortcutted functions.
# The third line removes `_rpc' from the names and reconstitutes the lines.
	( echo "#include <mach/mach_types.h>" ; \
	  echo "#include <mach/message.h>" ; \
	  echo ; \
	  cat $^ | tr \\012 @ | sed s/@@/@%/g | tr % \\012 \
	  | grep '^/\* Routine [a-z0-9_]*_rpc \*/' \
	  | sed 's/_rpc//g' | tr @ \\012 ) > $@-new
	mv -f $@-new $@
generated += mach-shortcuts.h

before-compile += $(objpfx)mach-shortcuts.h

include Machrules
include ../Rules


# A gcc bug prevents the generated file from working properly,
# so we have one in the distribution for the time being.
generated += errsystems.c
$(objpfx)errsystems.c: errsystems.awk err_*.sub \
		       $(wildcard $(addsuffix /err_*.sub,$(+sysdep_dirs)))
	$(AWK) -v subsys='$(filter-out $<,$^)' -f $^ > $@.n
	mv -f $@.n $@


# Convenient target to update all the generated source files.
.PHONY: generated
generated: $(addprefix $(common-objpfx),$(common-generated)) \
	   $(addprefix $(objpfx),$(generated))

.PHONY: mach-before-compile
mach-before-compile: $(mach-before-compile)
