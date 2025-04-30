# Used by *context() functions
ifeq ($(subdir),stdlib)
gen-as-const-headers += ucontext_i.sym
endif

ifeq ($(subdir),elf)
# Supporting non-executable stacks on HPPA requires changes to both
# the Linux kernel and glibc. The kernel currently needs an executable
# stack for syscall restarts and signal returns.
test-xfail-check-execstack = yes

# On hppa, the PLT is executable because it contains an executable
# trampoline used during lazy binding.
test-xfail-check-wx-segment = *

endif # $(subdir) == elf
