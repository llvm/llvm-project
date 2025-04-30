ifeq ($(subdir),csu)
gen-as-const-headers += cpu-features-offsets.sym
endif

ifeq ($(subdir),elf)
sysdep_routines += get-cpuid-feature-leaf
sysdep-dl-routines += dl-get-cpu-features
sysdep_headers += sys/platform/x86.h bits/platform/x86.h

CFLAGS-get-cpuid-feature-leaf.o += $(no-stack-protector)

tests += tst-get-cpu-features tst-get-cpu-features-static \
	 tst-cpu-features-cpuinfo tst-cpu-features-cpuinfo-static \
	 tst-cpu-features-supports tst-cpu-features-supports-static
tests-static += tst-get-cpu-features-static \
		tst-cpu-features-cpuinfo-static \
		tst-cpu-features-supports-static
ifeq (yes,$(have-ifunc))
ifeq (yes,$(have-gcc-ifunc))
tests += \
  tst-ifunc-isa-1 \
  tst-ifunc-isa-1-static
tests-static += \
  tst-ifunc-isa-1-static
ifneq ($(have-tunables),no)
tests += \
  tst-ifunc-isa-2 \
  tst-ifunc-isa-2-static
tests-static += \
  tst-ifunc-isa-2-static
endif
endif
endif
ifeq (yes,$(enable-x86-isa-level))
tests += tst-isa-level-1
modules-names += tst-isa-level-mod-1-baseline \
		 tst-isa-level-mod-1-v2 \
		 tst-isa-level-mod-1-v3 \
		 tst-isa-level-mod-1-v4 \

# X86 ISA level baseline
CFLAGS-tst-isa-level-mod-1-baseline.c += -DINCLUDE_X86_ISA_LEVEL \
					 -DISA_LEVEL=0x1 \
					 -march=x86-64
# X86 ISA level v2
CFLAGS-tst-isa-level-mod-1-v2.c += -DINCLUDE_X86_ISA_LEVEL \
				   -DISA_LEVEL=0x3 \
				   -march=x86-64
# X86 ISA level v3
CFLAGS-tst-isa-level-mod-1-v3.c += -DINCLUDE_X86_ISA_LEVEL \
				   -DISA_LEVEL=0x7 \
				   -march=x86-64
# X86 ISA level v4
CFLAGS-tst-isa-level-mod-1-v4.c += -DINCLUDE_X86_ISA_LEVEL \
				   -DISA_LEVEL=0xf \
				   -march=x86-64

$(objpfx)tst-isa-level-1.out: $(objpfx)tst-isa-level-mod-1-baseline.so \
			      $(objpfx)tst-isa-level-mod-1-v2.so \
			      $(objpfx)tst-isa-level-mod-1-v3.so \
			      $(objpfx)tst-isa-level-mod-1-v4.so
endif
ifneq ($(have-tunables),no)
tst-ifunc-isa-2-ENV = GLIBC_TUNABLES=glibc.cpu.hwcaps=-SSE4_2,-AVX,-AVX2,-AVX512F
tst-ifunc-isa-2-static-ENV = $(tst-ifunc-isa-2-ENV)
endif
endif

ifeq ($(subdir),math)
tests += tst-ldbl-nonnormal-printf
endif # $(subdir) == math

ifeq ($(subdir),setjmp)
gen-as-const-headers += jmp_buf-ssp.sym
sysdep_routines += __longjmp_cancel
ifneq ($(enable-cet),no)
ifneq ($(have-tunables),no)
tests += tst-setjmp-cet
tst-setjmp-cet-ENV = GLIBC_TUNABLES=glibc.cpu.x86_ibt=on:glibc.cpu.x86_shstk=on
endif
endif
endif

ifeq ($(subdir),string)
sysdep_routines += cacheinfo

tests += \
  tst-memchr-rtm \
  tst-memcmp-rtm \
  tst-memmove-rtm \
  tst-memrchr-rtm \
  tst-memset-rtm \
  tst-strchr-rtm \
  tst-strcpy-rtm \
  tst-strlen-rtm \
  tst-strncmp-rtm \
  tst-strrchr-rtm

CFLAGS-tst-memchr-rtm.c += -mrtm
CFLAGS-tst-memcmp-rtm.c += -mrtm
CFLAGS-tst-memmove-rtm.c += -mrtm
CFLAGS-tst-memrchr-rtm.c += -mrtm
CFLAGS-tst-memset-rtm.c += -mrtm
CFLAGS-tst-strchr-rtm.c += -mrtm
CFLAGS-tst-strcpy-rtm.c += -mrtm
CFLAGS-tst-strlen-rtm.c += -mrtm
CFLAGS-tst-strncmp-rtm.c += -mrtm
CFLAGS-tst-strrchr-rtm.c += -mrtm
endif

ifneq ($(enable-cet),no)
ifeq ($(subdir),elf)
sysdep-dl-routines += dl-cet

tests += tst-cet-legacy-1 tst-cet-legacy-1a tst-cet-legacy-2 \
	 tst-cet-legacy-2a tst-cet-legacy-3 tst-cet-legacy-4 \
	 tst-cet-legacy-5a tst-cet-legacy-6a tst-cet-legacy-7 \
	 tst-cet-legacy-8 tst-cet-legacy-9 tst-cet-legacy-9-static \
	 tst-cet-legacy-10 tst-cet-legacy-10-static
tests-static += tst-cet-legacy-9-static tst-cet-legacy-10-static
tst-cet-legacy-1a-ARGS = -- $(host-test-program-cmd)
ifneq (no,$(have-tunables))
tests += tst-cet-legacy-4a tst-cet-legacy-4b tst-cet-legacy-4c \
	 tst-cet-legacy-5b tst-cet-legacy-6b
endif
modules-names += tst-cet-legacy-mod-1 tst-cet-legacy-mod-2 \
		 tst-cet-legacy-mod-4 tst-cet-legacy-mod-5a \
		 tst-cet-legacy-mod-5b tst-cet-legacy-mod-5c \
		 tst-cet-legacy-mod-6a tst-cet-legacy-mod-6b \
		 tst-cet-legacy-mod-6c

CFLAGS-tst-cet-legacy-2.c += -fcf-protection=branch
CFLAGS-tst-cet-legacy-2a.c += -fcf-protection
CFLAGS-tst-cet-legacy-mod-1.c += -fcf-protection=none
CFLAGS-tst-cet-legacy-mod-2.c += -fcf-protection=none
CFLAGS-tst-cet-legacy-3.c += -fcf-protection=none
CFLAGS-tst-cet-legacy-4.c += -fcf-protection=branch
CFLAGS-tst-cet-legacy-4a.c += -fcf-protection
CFLAGS-tst-cet-legacy-4b.c += -fcf-protection
CFLAGS-tst-cet-legacy-mod-4.c += -fcf-protection=none
CFLAGS-tst-cet-legacy-5a.c += -fcf-protection -mshstk
ifeq ($(enable-cet),permissive)
CPPFLAGS-tst-cet-legacy-5a.c += -DCET_IS_PERMISSIVE=1
endif
CFLAGS-tst-cet-legacy-5b.c += -fcf-protection -mshstk
CPPFLAGS-tst-cet-legacy-5b.c += -DCET_DISABLED_BY_ENV=1
CFLAGS-tst-cet-legacy-mod-5a.c += -fcf-protection=branch
CFLAGS-tst-cet-legacy-mod-5b.c += -fcf-protection
CFLAGS-tst-cet-legacy-mod-5c.c += -fcf-protection
CFLAGS-tst-cet-legacy-6a.c += -fcf-protection -mshstk
ifeq ($(enable-cet),permissive)
CPPFLAGS-tst-cet-legacy-6a.c += -DCET_IS_PERMISSIVE=1
endif
CFLAGS-tst-cet-legacy-6b.c += -fcf-protection -mshstk
CPPFLAGS-tst-cet-legacy-6b.c += -DCET_DISABLED_BY_ENV=1
CFLAGS-tst-cet-legacy-mod-6a.c += -fcf-protection=branch
CFLAGS-tst-cet-legacy-mod-6b.c += -fcf-protection
CFLAGS-tst-cet-legacy-mod-6c.c += -fcf-protection
CFLAGS-tst-cet-legacy-7.c += -fcf-protection=none
CFLAGS-tst-cet-legacy-8.c += -mshstk
CFLAGS-tst-cet-legacy-10.c += -mshstk
CFLAGS-tst-cet-legacy-10-static.c += -mshstk

$(objpfx)tst-cet-legacy-1: $(objpfx)tst-cet-legacy-mod-1.so \
		       $(objpfx)tst-cet-legacy-mod-2.so
$(objpfx)tst-cet-legacy-1a: $(objpfx)tst-cet-legacy-mod-1.so \
		       $(objpfx)tst-cet-legacy-mod-2.so
$(objpfx)tst-cet-legacy-2: $(objpfx)tst-cet-legacy-mod-2.so
$(objpfx)tst-cet-legacy-2.out: $(objpfx)tst-cet-legacy-mod-1.so
$(objpfx)tst-cet-legacy-2a: $(objpfx)tst-cet-legacy-mod-2.so
$(objpfx)tst-cet-legacy-2a.out: $(objpfx)tst-cet-legacy-mod-1.so
$(objpfx)tst-cet-legacy-4.out: $(objpfx)tst-cet-legacy-mod-4.so
$(objpfx)tst-cet-legacy-5a.out: $(objpfx)tst-cet-legacy-mod-5a.so \
				$(objpfx)tst-cet-legacy-mod-5b.so
$(objpfx)tst-cet-legacy-mod-5a.so: $(objpfx)tst-cet-legacy-mod-5c.so
$(objpfx)tst-cet-legacy-mod-5b.so: $(objpfx)tst-cet-legacy-mod-5c.so
$(objpfx)tst-cet-legacy-6a.out: $(objpfx)tst-cet-legacy-mod-6a.so \
				$(objpfx)tst-cet-legacy-mod-6b.so
$(objpfx)tst-cet-legacy-mod-6a.so: $(objpfx)tst-cet-legacy-mod-6c.so
$(objpfx)tst-cet-legacy-mod-6b.so: $(objpfx)tst-cet-legacy-mod-6c.so
LDFLAGS-tst-cet-legacy-mod-6c.so = -Wl,--enable-new-dtags,-z,nodelete
ifneq (no,$(have-tunables))
$(objpfx)tst-cet-legacy-4a.out: $(objpfx)tst-cet-legacy-mod-4.so
tst-cet-legacy-4a-ENV = GLIBC_TUNABLES=glibc.cpu.x86_shstk=permissive
$(objpfx)tst-cet-legacy-4b.out: $(objpfx)tst-cet-legacy-mod-4.so
tst-cet-legacy-4b-ENV = GLIBC_TUNABLES=glibc.cpu.x86_shstk=on
$(objpfx)tst-cet-legacy-4c.out: $(objpfx)tst-cet-legacy-mod-4.so
tst-cet-legacy-4c-ENV = GLIBC_TUNABLES=glibc.cpu.x86_shstk=off
$(objpfx)tst-cet-legacy-5b.out: $(objpfx)tst-cet-legacy-mod-5a.so \
				$(objpfx)tst-cet-legacy-mod-5b.so
tst-cet-legacy-5b-ENV = GLIBC_TUNABLES=glibc.cpu.hwcaps=-IBT,-SHSTK
$(objpfx)tst-cet-legacy-6b.out: $(objpfx)tst-cet-legacy-mod-6a.so \
				$(objpfx)tst-cet-legacy-mod-6b.so
tst-cet-legacy-6b-ENV = GLIBC_TUNABLES=glibc.cpu.hwcaps=-IBT,-SHSTK
tst-cet-legacy-9-ENV = GLIBC_TUNABLES=glibc.cpu.hwcaps=-IBT,-SHSTK
tst-cet-legacy-9-static-ENV = GLIBC_TUNABLES=glibc.cpu.hwcaps=-IBT,-SHSTK
endif
endif

# Add -fcf-protection to CFLAGS when CET is enabled.
CFLAGS-.o += -fcf-protection
CFLAGS-.os += -fcf-protection
CFLAGS-.op += -fcf-protection
CFLAGS-.oS += -fcf-protection

# Compile assembly codes with <cet.h> when CET is enabled.
asm-CPPFLAGS += -fcf-protection -include cet.h

ifeq ($(subdir),elf)
ifeq (yes,$(build-shared))
tests-special += $(objpfx)check-cet.out
endif

# FIXME: Can't use all-built-dso in elf/Makefile since this file is
# processed before elf/Makefile.  Duplicate it here.
cet-built-dso := $(common-objpfx)elf/ld.so $(common-objpfx)libc.so \
		 $(filter-out $(common-objpfx)linkobj/libc.so, \
			      $(sort $(wildcard $(addprefix $(common-objpfx), \
							    */lib*.so \
							    iconvdata/*.so))))

$(cet-built-dso:=.note): %.note: %
	@rm -f $@T
	LC_ALL=C $(READELF) -n $< > $@T
	test -s $@T
	mv -f $@T $@
common-generated += $(cet-built-dso:$(common-objpfx)%=%.note)

$(objpfx)check-cet.out: $(..)sysdeps/x86/check-cet.awk \
			$(cet-built-dso:=.note)
	LC_ALL=C $(AWK) -f $^ > $@; \
	$(evaluate-test)
generated += check-cet.out
endif
endif

ifeq ($(subdir),posix)
tests += \
  tst-sysconf-cache-linesize \
  tst-sysconf-cache-linesize-static
tests-static += \
  tst-sysconf-cache-linesize-static
endif
