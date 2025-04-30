ifeq ($(subdir),misc)
sysdep_routines += ioperm iopl mremap
endif

ifeq ($(subdir),stdlib)
sysdep_routines += __start_context
endif

ifeq ($(subdir),csu)
gen-as-const-headers += ucontext_i.sym
endif

ifeq ($(subdir),misc)
gen-as-const-headers += sigaltstack-offsets.sym
endif

ifeq ($(subdir),elf)
ifeq (yes,$(enable-x86-isa-level))
tests += \
  tst-glibc-hwcaps-2
ifeq (no,$(build-hardcoded-path-in-tests))
# This is an ld.so.cache test, and RPATH/RUNPATH in the executable
# interferes with its test objectives.
tests-container += \
  tst-glibc-hwcaps-2-cache
endif
modules-names += \
  libx86-64-isa-level-1 \
  libx86-64-isa-level-2 \
  libx86-64-isa-level-3 \
  libx86-64-isa-level-4

$(objpfx)tst-glibc-hwcaps-2: $(objpfx)libx86-64-isa-level.so

$(objpfx)tst-glibc-hwcaps-2.out: \
  $(objpfx)glibc-hwcaps/x86-64-v2/libx86-64-isa-level.so \
  $(objpfx)glibc-hwcaps/x86-64-v4/libx86-64-isa-level.so \
  $(objpfx)glibc-hwcaps/x86-64-v3/libx86-64-isa-level.so
$(objpfx)glibc-hwcaps/x86-64-v2/libx86-64-isa-level.so: \
  $(objpfx)libx86-64-isa-level-2.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/x86-64-v3/libx86-64-isa-level.so: \
  $(objpfx)libx86-64-isa-level-3.so
	$(make-target-directory)
	cp $< $@
$(objpfx)glibc-hwcaps/x86-64-v4/libx86-64-isa-level.so: \
  $(objpfx)libx86-64-isa-level-4.so
	$(make-target-directory)
	cp $< $@

CFLAGS-libx86-64-isa-level-1.os += -march=x86-64
CFLAGS-libx86-64-isa-level-2.os += -march=x86-64
CFLAGS-libx86-64-isa-level-3.os += -march=x86-64
CFLAGS-libx86-64-isa-level-4.os += -march=x86-64

# The test modules are parameterized by preprocessor macros.
LDFLAGS-libx86-64-isa-level-1.so += -Wl,-soname,libx86-64-isa-level.so
LDFLAGS-libx86-64-isa-level-4.so += -Wl,-soname,libx86-64-isa-level.so
$(objpfx)libx86-64-isa-level%.os: $(..)/sysdeps/unix/sysv/linux/x86_64/x86-64-isa-level-VALUE.c
	$(compile-command.c) -DVALUE=$(lastword $(subst -, ,$*)) \
	  -DISA_LEVEL="(1 << ($(lastword $(subst -, ,$*)) - 1))"
$(objpfx)libx86-64-isa-level.so: $(objpfx)libx86-64-isa-level-1.so
	cp $< $@
endif
endif # $(subdir) == elf
