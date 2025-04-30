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

ifndef inhibit-unix-syscalls

# Sysdep dirs unix/... can contain a file syscalls.list,
# which specifies objects to be compiled as simple Unix system calls.

-include $(common-objpfx)sysd-syscalls

ifeq (misc,$(subdir))
sysdep_routines += $(unix-extra-syscalls)

ifdef unix-stub-syscalls
# The system call entry points in this list are supposed to be additional
# functions not overriding any other sysdeps/.../call.c implementation, but
# their system call numbers are unavailable in the kernel headers we're
# using.  Instead of a system call stub, these get a function that fails
# with ENOSYS.  We just generate a single module defining one function and
# making all these entry point names aliases for it.
sysdep_routines += stub-syscalls
$(objpfx)stub-syscalls.c: $(common-objpfx)sysd-syscalls \
			  $(..)sysdeps/unix/Makefile
	$(make-target-directory)
	(for call in $(unix-stub-syscalls); do \
	   call=$${call%%@*}; \
	   echo "#define $$call RENAMED_$$call"; \
	 done; \
	 echo '#include <errno.h>'; \
	 echo '#include <shlib-compat.h>'; \
	 for call in $(unix-stub-syscalls); do \
	   call=$${call%%@*}; \
	   echo "#undef $$call"; \
	 done; \
	 echo 'long int _no_syscall (void)'; \
	 echo '{ __set_errno (ENOSYS); return -1L; }'; \
	 for call in $(unix-stub-syscalls); do \
	   case $$call in \
	   *@@*) \
	     ver=$${call##*@}; call=$${call%%@*}; \
	     ver=`echo "$$ver" | sed 's/\./_/g'`; \
	     echo "strong_alias (_no_syscall, __$${call}_$${ver})"; \
	     echo "versioned_symbol (libc, __$${call}_$${ver}, $$call, $$ver);"\
	     ;; \
	   *@*) \
	     ver=$${call##*@}; call=$${call%%@*}; \
	     ver=`echo "$$ver" | sed 's/\./_/g'`; \
	     case $$ver in \
	     *:*) \
	       compat_ver=$${ver#*:}; \
	       ver=$${ver%%:*}; \
	       compat_cond="SHLIB_COMPAT (libc, $$ver, $$compat_ver)"; \
	       ;; \
	     *) \
	       compat_cond=""; \
	       ;; \
	     esac; \
	     if [ -n "$$compat_cond" ]; then echo "#if $$compat_cond"; fi; \
	     echo "strong_alias (_no_syscall, __$${call}_$${ver})"; \
	     echo "compat_symbol (libc, __$${call}_$${ver}, $$call, $$ver);"; \
	     if [ -n "$$compat_cond" ]; then echo "#endif"; fi; \
	     ;; \
	   *) echo "weak_alias (_no_syscall, $$call)"; \
	      echo "stub_warning ($$call)"; \
	      echo "weak_alias (_no_syscall, __GI_$$call)" ;; \
	   esac; \
	 done) > $@T
	mv -f $@T $@
generated += stub-syscalls.c
endif
endif

# This is the end of the pipeline for compiling the syscall stubs.
# The stdin is assembler with cpp using sysdep.h macros.
compile-syscall = $(COMPILE.S) -o $@ -x assembler-with-cpp - \
			       $(compile-mkdep-flags)

ifndef avoid-generated
$(common-objpfx)sysd-syscalls: $(..)sysdeps/unix/make-syscalls.sh \
			       $(wildcard $(+sysdep_dirs:%=%/syscalls.list)) \
			       $(wildcard $(+sysdep_dirs:%=%/arch-syscall.h)) \
			       $(common-objpfx)libc-modules.stmp
	for dir in $(+sysdep_dirs); do \
	  test -f $$dir/syscalls.list && \
	  { sysdirs='$(sysdirs)' \
	    asm_CPP='$(COMPILE.S) -E -x assembler-with-cpp' \
	    $(SHELL) $(dir $<)$(notdir $<) $$dir || exit 1; }; \
	  test $$dir = $(..)sysdeps/unix && break; \
	done > $@T
	mv -f $@T $@
endif

$(common-objpfx)s-%.d: $(..)sysdeps/unix/s-%.S \
		       $(wildcard $(+sysdep_dirs:%=%/syscalls.list))
	$(+make-deps)

postclean-generated += sysd-syscalls

endif
