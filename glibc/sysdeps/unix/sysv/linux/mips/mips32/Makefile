ifeq ($(subdir),conform)
# For bugs 17786 and 21278.
conformtest-xfail-conds += mips-o32-linux
endif

ifeq ($(subdir),misc)
sysdep_routines += mips-syscall5 mips-syscall6 mips-syscall7
endif

ifeq ($(subdir),stdlib)
tests += bug-getcontext-mips-gp
endif
