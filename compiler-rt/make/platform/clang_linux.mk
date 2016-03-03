Description := Static runtime libraries for clang/Linux.

###

CC := clang
Arch := unknown
Configs :=

# We don't currently have any general purpose way to target architectures other
# than the compiler defaults (because there is no generalized way to invoke
# cross compilers). For now, we just find the target architecture of the
# compiler and only define configurations we know that compiler can generate.
CompilerTargetTriple := $(shell \
	LANG=C $(CC) -v 2>&1 | grep 'Target:' | cut -d' ' -f2)
ifeq ($(CompilerTargetTriple),)
$(error "unable to infer compiler target triple for $(CC)")
endif

# Only define configs if we detected a linux target.
ifneq ($(findstring -linux-,$(CompilerTargetTriple)),)

# Define configs only if arch in triple is i386 or x86_64
CompilerTargetArch := $(firstword $(subst -, ,$(CompilerTargetTriple)))
ifeq ($(call contains,i386 x86_64,$(CompilerTargetArch)),true)

# TryCompile compiler source flags
# Returns exit code of running a compiler invocation.
TryCompile = \
  $(shell \
    cflags=""; \
    for flag in $(3); do \
      cflags="$$cflags $$flag"; \
    done; \
    $(1) $$cflags $(2) -o /dev/null > /dev/null 2> /dev/null ; \
    echo $$?)

test_source = $(ProjSrcRoot)/make/platform/clang_linux_test_input.c
ifeq ($(CompilerTargetArch),i386)
  SupportedArches := i386
  ifeq ($(call TryCompile,$(CC),$(test_source),-m64),0)
    SupportedArches += x86_64
  endif
else
  SupportedArches := x86_64
  ifeq ($(call TryCompile,$(CC),$(test_source),-m32),0)
    SupportedArches += i386
  endif
endif

# Build runtime libraries for i386.
ifeq ($(call contains,$(SupportedArches),i386),true)
Configs += builtins-i386 profile-i386
Arch.builtins-i386 := i386
Arch.profile-i386 := i386
endif

# Build runtime libraries for x86_64.
ifeq ($(call contains,$(SupportedArches),x86_64),true)
Configs += builtins-x86_64 profile-x86_64
Arch.builtins-x86_64 := x86_64
Arch.profile-x86_64 := x86_64
endif

endif

endif

###

CFLAGS := -Wall -Werror -O3 -fomit-frame-pointer

CFLAGS.builtins-i386 := $(CFLAGS) -m32
CFLAGS.builtins-x86_64 := $(CFLAGS) -m64
CFLAGS.profile-i386 := $(CFLAGS) -m32
CFLAGS.profile-x86_64 := $(CFLAGS) -m64

FUNCTIONS.builtins-i386 := $(CommonFunctions) $(ArchFunctions.i386)
FUNCTIONS.builtins-x86_64 := $(CommonFunctions) $(ArchFunctions.x86_64)
FUNCTIONS.profile-i386 := GCDAProfiling InstrProfiling InstrProfilingBuffer \
                          InstrProfilingFile InstrProfilingPlatformOther \
                          InstrProfilingRuntime InstrProfilingUtil \
                          InstrProfilingWriter InstrProfilingValue \
                          InstrProfilingMerge InstrProfilingMergeFile
FUNCTIONS.profile-x86_64 := $(FUNCTIONS.profile-i386)

# Always use optimized variants.
OPTIMIZED := 1

# We don't need to use visibility hidden on Linux.
VISIBILITY_HIDDEN := 0

SHARED_LIBRARY_SUFFIX := so
