ifeq ($(subdir),elf)
sysdep-rtld-routines += aeabi_read_tp libc-do-syscall
endif

ifeq ($(subdir),misc)
sysdep_routines += ioperm
sysdep_headers += sys/elf.h
endif

ifeq ($(subdir),signal)
sysdep_routines += sigrestorer
endif

ifeq ($(subdir),stdlib)
gen-as-const-headers += ucontext_i.sym
endif

# Add a syscall function to each library that needs one.

ifeq ($(subdir),crypt)
libcrypt-sysdep_routines += libc-do-syscall
endif

ifeq ($(subdir),rt)
librt-sysdep_routines += libc-do-syscall
librt-shared-only-routines += libc-do-syscall
endif

ifeq ($(subdir),resolv)
libanl-sysdep_routines += libc-do-syscall
libanl-shared-only-routines += libc-do-syscall
endif

ifeq ($(subdir),csu)
sysdep_routines += libc-do-syscall
endif

ifeq ($(subdir),nscd)
nscd-modules += libc-do-syscall
endif

ifeq ($(subdir),nss)
libnss_db-sysdep_routines += libc-do-syscall
libnss_db-shared-only-routines += libc-do-syscall
endif

ifeq ($(subdir),posix)
LDFLAGS-tst-rfc3484 += $(common-objpfx)csu/libc-do-syscall.o
LDFLAGS-tst-rfc3484-2 += $(common-objpfx)csu/libc-do-syscall.o
LDFLAGS-tst-rfc3484-3 += $(common-objpfx)csu/libc-do-syscall.o
endif

abi-variants := soft hard

ifeq (,$(filter $(default-abi),$(abi-variants)))
Unknown ABI, must be one of $(abi-variants)
endif

abi-includes :=
abi-soft-condition := !defined __ARM_PCS_VFP
abi-hard-condition := defined __ARM_PCS_VFP
