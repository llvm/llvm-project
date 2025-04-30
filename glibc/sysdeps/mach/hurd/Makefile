# Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

ifdef in-Makerules

# Look for header files in hurd/ under the top-level library source directory.
# Look for generated header files where they get created.
includes += -I$(..)hurd -I$(common-objpfx)hurd/

# We use the style `if (err = call(...))' a lot in the Hurd code,
# where we have a lot of functions that return zero or an errno code.
+cflags += -Wno-parentheses

# Do not use any assembly code from sysdeps/unix (and subdirectories).
# This bypasses all the system call stubs and uses any existing posix or
# generic C files instead.
inhibit-sysdep-asm += unix.*
inhibit-unix-syscalls = yes

# Don't try to generate anything from the installed Unix system and its
# libraries.  That is only of use when building for a Unix system, so as to
# be compatible with some existing binaries for that system.
inhibit-glue = yes

ifeq (,$(filter mach hurd,$(subdir)))
# Subdirectories other than hurd/ might use the generated Hurd headers.
# So make sure we get a chance to run in hurd/ to make them before all else.
# (But we don't want to do this in mach/, because hurd/ needs some things
# there, and we know mach/ doesn't need anything from hurd/.)

hurd-objpfx = $(common-objpfx)hurd/

# These are all the generated headers that <hurd.h> includes.
before-compile += $(patsubst %,$(hurd-objpfx)hurd/%.h,auth io fs process)
$(patsubst %,$(hurd-objpfx)hurd/%.h,auth io fs process): hurd-before-compile
.PHONY: hurd-before-compile
hurd-before-compile: $(common-objpfx)mach/mach-shortcuts.h
	$(MAKE) -C $(..)hurd subdir=hurd before-compile no_deps=t
endif

# Hurd profil.c includes this file, so give a rule to make it.
ifeq ($(subdir),gmon)
$(common-objpfx)hurd/../mach/RPC_task_get_sampled_pcs.c:
	$(MAKE) -C $(..)mach subdir=mach before-compile no_deps=t
endif


# Generate bits/errno.h from the section of the manual that lists all the errno
# codes.

errno.texinfo = $(..)manual/errno.texi

hurd = $(..)sysdeps/mach/hurd

define mach-errno-h
($(foreach h,mach/message.h \
	     mach/kern_return.h \
	     mach/mig_errors.h \
	     device/device_types.h,\
 echo '#include <$h>';\
 ))
endef

# We use the compiler to generate a list of absolute file names for
# the headers we want to search for Mach error codes, listed above (and
# incidentally, all other headers those include).
-include $(common-objpfx)errnos.d
$(common-objpfx)errnos.d: $(mach-errnos-deps)
	$(mach-errno-h) | \
	$(CC) $(CFLAGS) \
	    $(subst -include $(common-objpfx)libc-modules.h,,$(CPPFLAGS)) \
	    -M -x c - | \
	sed $(sed-remove-objpfx) -e 's,- *:,mach-errnos-deps :=,' \
	    -e 's, \.\./, $(..),g' > $@t
	mv -f $@t $@

$(hurd)/bits/errno.h: $(common-objpfx)stamp-errnos ;
$(common-objpfx)stamp-errnos: $(hurd)/errnos.awk $(errno.texinfo) \
			      $(mach-errnos-deps) $(common-objpfx)errnos.d
	mkdir -p $(hurd-objpfx)bits
	$(AWK) -f $^ > $(hurd-objpfx)bits/errno.h-tmp
# Make it unwritable so noone will edit it by mistake.
	-chmod a-w $(hurd-objpfx)bits/errno.h-tmp
	$(move-if-change) $(hurd-objpfx)bits/errno.h-tmp $(hurd)/bits/errno.h
	touch $@

common-generated += errnos.d stamp-errnos

# We install the real libc.a as libcrt.a and as libc.a we install a linker
# script which does -( -lcrt -lmachuser -lhurduser -).

libc-name = crt

ifeq (,$(subdir))
install-others += $(inst_libdir)/libc.a
$(inst_libdir)/libc.a: $(hurd)/libc-ldscript $(+force); $(do-install)
ifeq (yes,$(build-profile))
install-others += $(inst_libdir)/libc_p.a
$(inst_libdir)/libc_p.a: $(hurd)/libc_p-ldscript $(+force); $(do-install)
endif
endif

# Make sure these are used to build the libc.so shared object too.  There
# is a circular dependency between each of these shared objects and libc
# (many high-level libc functions call stubs, stubs call low-level libc
# functions like memcpy and mach_msg).  This works out fine at run time
# (all the objects are loaded before resolving their symbols, so these
# interdependencies are fine).  But to create the shared objects we must
# link them one at a time; since each needs one or both of the others to
# produce its DT_NEEDED entries and to assign its undefined symbols the
# right symbol versions, we can't do any of them before the others!  To
# get around this, we link each lib*user.so shared object twice.  First,
# we link an object without reference to libc.so (since we haven't linked
# libc.so yet), so it lacks a DT_NEEDED record for the libc soname it
# depends on, and its undefined symbol references lack the symbol version
# assignments they should have.  We will use this shared object solely to
# link libc.so against it; that gives libc.so the proper DT_NEEDED record,
# and symbol versions assignments (if the lib*user.so object is using them).
# Finally we link a second version of the same lib*user.so shared object,
# this time linked normally against libc so it gets a proper DT_NEEDED
# record and symbol version set; this one can be installed for run-time use.
rpcuserlibs := $(common-objpfx)mach/libmachuser.so \
	       $(common-objpfx)hurd/libhurduser.so
link-rpcuserlibs := $(rpcuserlibs:%user.so=%user-link.so)
$(common-objpfx)libc.so: $(link-rpcuserlibs)
$(common-objpfx)linkobj/libc.so: $(link-rpcuserlibs)
rpath-dirs += mach hurd

# Make sure the `lib' pass builds the dummy shared objects so
# we can link libc against them.
ifeq (mach,$(subdir))
lib-noranlib: $(common-objpfx)mach/libmachuser-link.so
endif
ifeq (hurd,$(subdir))
lib-noranlib: $(common-objpfx)hurd/libhurduser-link.so
endif

$(link-rpcuserlibs): %-link.so: %_pic.a
# These shared objects are just for the purpose of linking libc,
# so they don't need abi-note.o linked into them.
	$(build-shlib-helper) \
		-nostdlib -o $@ \
		-Wl,-soname=$(*F).so$($(*F).so-version) \
		$(build-shlib-objlist)
libmachuser-link.so-no-z-defs = yes
libhurduser-link.so-no-z-defs = yes

# And get them into the libc.so ldscript.
$(inst_libdir)/libc.so: $(rpcuserlibs)

# The RPC stubs from these libraries are needed in building the dynamic
# linker, too.  It must be self-contained, so we link the needed PIC
# objects directly into the shared object.
ifeq (elf,$(subdir))
$(objpfx)librtld.map: $(rpcuserlibs:.so=_pic.a)

CFLAGS-dl-load.c = -DEXTERNAL_MAP_FROM_FD
endif

# Override the generic Makeconfig values so we link against the RPC libs.
link-libc-static := -Wl,--start-group \
		    $(patsubst %,$(common-objpfx)%.a,\
			       libc mach/libmachuser hurd/libhurduser) \
		    $(static-gnulib) -Wl,--end-group
link-libc-static-tests := -Wl,--start-group \
		    $(patsubst %,$(common-objpfx)%.a,\
			       libc mach/libmachuser hurd/libhurduser) \
		    $(static-gnulib-tests) -Wl,--end-group

ifeq ($(subdir),csu)

extra-objs += static-start.o

# We need special startup code for statically linked binaries.
$(objpfx)crt0.o: $(objpfx)static-start.o $(objpfx)abi-note.o $(objpfx)init.o
	$(link-relocatable)

endif

ifeq (hurd, $(subdir))
sysdep_routines += cthreads
endif

ifeq (io, $(subdir))
sysdep_routines += f_setlk close_nocancel close_nocancel_nostatus \
		   fcntl_nocancel open_nocancel openat_nocancel read_nocancel \
		   pread64_nocancel write_nocancel pwrite64_nocancel \
		   wait4_nocancel \
		   xstat fxstat lxstat xstat64 fxstat64 lxstat64 \
		   xmknod xmknodat \
		   fxstatat fxstatat64
endif

ifeq (misc, $(subdir))
sysdep_routines += writev_nocancel writev_nocancel_nostatus mremap
endif

ifeq ($(subdir),sunrpc)
sysdep_headers += nfs/nfs.h
endif

ifeq ($(subdir),socket)
sysdep_headers += net/ethernet.h net/if_arp.h net/if_ether.h net/route.h
endif

ifeq ($(subdir),nis)
CFLAGS-ypclnt.c += -DUSE_BINDINGDIR=1
endif

LDLIBS-pthread.so += $(objdir)/hurd/libhurduser.so
LDLIBS-c_malloc_debug.so += $(link-rpcuserlibs)

endif	# in-Makerules
