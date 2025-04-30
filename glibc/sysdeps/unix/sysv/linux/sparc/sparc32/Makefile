# See Makeconfig regarding the use of default-abi.
default-abi := 32

asm-CPPFLAGS = -D__ASSEMBLY__
ASFLAGS-.os += -fPIC
LD += -melf32_sparc

ifeq ($(subdir),stdlib)
gen-as-const-headers += ucontext_i.sym
endif

# When I get this to work, this is the right thing
ifeq ($(subdir),elf)
CFLAGS-rtld.c += -mcpu=v8
#rtld-routines += dl-sysdepsparc
sysdep-others += lddlibc4
install-bin += lddlibc4
endif   # elf

ifeq ($(subdir),math)
# These 2 routines are normally in libgcc{.a,_s.so.1}.
# However, sparc32 -mlong-double-128 libgcc relies on
# glibc providing _Q_* routines and without these files
# glibc relies on __multc3/__divtc3 only provided
# by libgcc if configured with -mlong-double-128.
# Provide these routines here as well.
libm-routines += multc3 divtc3
endif   # math
