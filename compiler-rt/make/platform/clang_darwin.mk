# These are the functions which clang needs when it is targeting a previous
# version of the OS. The issue is that the backend may use functions which were
# not present in the libgcc that shipped on the platform. In such cases, we link
# with a version of the library which contains private_extern definitions of all
# the extra functions which might be referenced.

Description := Static runtime libraries for clang/Darwin.

# A function that ensures we don't try to build for architectures and SDKs
# that we don't have working toolchains for. Arguments:
# (1): List of architectures
# (2): Library name
# (3): SDK path
# The result is a possibly empty subset of the architectures from argument 1.
CheckArches = \
  $(shell \
    result=""; \
    if [ "X$(3)" != X ]; then \
      for arch in $(1); do \
        if $(LD) -v 2>&1 | grep "configured to support" \
             | tr ' ' '\n' | grep "^$$arch$$" >/dev/null 2>/dev/null; then \
          if $(CC) -arch $$arch \
            -integrated-as \
            $(ProjSrcRoot)/make/platform/clang_darwin_test_input.c \
            -isysroot $(3) \
            -o /dev/null > /dev/null 2> /dev/null; then \
              result="$$result$$arch "; \
          else \
            printf 1>&2 \
             "warning: clang_darwin.mk: dropping arch '$$arch' from lib '$(2)'"; \
            printf 1>&2 " (clang or system libraries do not support it)\n"; \
          fi; \
        else \
          printf 1>&2 \
            "warning: clang_darwin.mk: dropping arch '$$arch' from lib '$(2)'";\
          printf 1>&2 " (ld does not support it)\n"; \
        fi; \
      done; \
    fi; \
    echo $$result)

XCRun = \
  $(shell \
    result=`xcrun -find $(1) 2> /dev/null`; \
    if [ "$$?" != "0" ]; then result=$(1); fi; \
    echo $$result)
# Prefer building with the internal SDKs.
XCRunSdkPath = \
  $(shell \
    result=`xcrun --sdk $(1).internal --show-sdk-path 2> /dev/null`; \
    if [ "$$?" != "0" ]; then \
      result=`xcrun --sdk $(1) --show-sdk-path 2> /dev/null`; \
      if [ "$$?" != "0" ]; then result=""; fi; \
    fi; \
    echo $$result)
###

CC       := $(call XCRun,clang)
LD       := $(shell $(CC) -print-prog-name=ld)
AR       := $(call XCRun,ar)
RANLIB   := $(call XCRun,ranlib)
STRIP    := $(call XCRun,strip)
LIPO     := $(call XCRun,lipo)
DSYMUTIL := $(call XCRun,dsymutil)

OSX_SDK := $(call XCRunSdkPath,macosx)
IOS_SDK := $(call XCRunSdkPath,iphoneos)
IOSSIM_SDK := $(call XCRunSdkPath,iphonesimulator)

Configs :=
UniversalArchs :=

# Configuration solely for providing access to an eprintf symbol, which may
# still be referenced from Darwin system headers. This symbol is only ever
# needed on i386.
Configs += eprintf
UniversalArchs.eprintf := $(call CheckArches,i386,eprintf,$(OSX_SDK))

# Configuration for targeting 10.4. We need a few functions missing from
# libgcc_s.10.4.dylib. We only build x86 slices since clang doesn't really
# support targeting PowerPC.
Configs += 10.4
UniversalArchs.10.4 := $(call CheckArches,i386 x86_64,10.4,$(OSX_SDK))

# Configuration for targeting iOS for a couple of functions that didn't
# make it into libSystem.
Configs += ios
UniversalArchs.ios := $(call CheckArches,i386 x86_64,ios,$(IOSSIM_SDK))
UniversalArchs.ios += $(call CheckArches,armv7 arm64,ios,$(IOS_SDK))

# Configuration for targeting OSX. These functions may not be in libSystem
# so we should provide our own.
Configs += osx
UniversalArchs.osx := $(call CheckArches,i386 x86_64 x86_64h,osx,$(OSX_SDK))

# Configuration for use with kernel/kexts.
Configs += cc_kext
UniversalArchs.cc_kext := $(call CheckArches,i386 x86_64 x86_64h,cc_kext,$(OSX_SDK))

# Configuration for use with iOS kernel/kexts
Configs += cc_kext_ios
UniversalArchs.cc_kext_ios += $(call CheckArches,armv7,cc_kext_ios,$(IOS_SDK))

# Darwin 10.6 has a bug in cctools that makes it unable to use ranlib on our ARM
# object files. If we are on that platform, strip out all ARM archs. We still
# build the libraries themselves so that Clang can find them where it expects
# them, even though they might not have an expected slice.
ifneq ($(shell test -x /usr/bin/sw_vers && sw_vers -productVersion | grep 10.6),)
UniversalArchs.ios := $(filter-out armv7, $(UniversalArchs.ios))
UniversalArchs.cc_kext_ios := $(filter-out armv7, $(UniversalArchs.cc_kext_ios))
endif

# If RC_SUPPORTED_ARCHS is defined, treat it as a list of the architectures we
# are intended to support and limit what we try to build to that.
ifneq ($(RC_SUPPORTED_ARCHS),)
$(foreach config,$(Configs),\
  $(call Set,UniversalArchs.$(config),\
	$(filter $(RC_SUPPORTED_ARCHS),$(UniversalArchs.$(config)))))
endif

# Remove empty configs if we end up dropping all the requested
# archs for a particular config.
$(foreach config,$(Configs),\
  $(if $(strip $(UniversalArchs.$(config))),,\
	$(call Set,Configs,$(filter-out $(config),$(Configs)))))

###

# Forcibly strip off any -arch, as that totally breaks our universal support.
override CC := $(subst -arch ,-arch_,$(CC))
override CC := $(patsubst -arch_%,,$(CC))

CFLAGS := -Wall -Werror -O3 -fomit-frame-pointer

# Always set deployment target arguments for every build, these libraries should
# never depend on the environmental overrides. We simply set them to minimum
# supported deployment target -- nothing in the compiler-rt libraries should
# actually depend on the deployment target.
OSX_DEPLOYMENT_ARGS := -mmacosx-version-min=10.4
IOS_DEPLOYMENT_ARGS := -mios-version-min=1.0
IOS6_DEPLOYMENT_ARGS := -mios-version-min=6.0
IOSSIM_DEPLOYMENT_ARGS := -mios-simulator-version-min=1.0

OSX_DEPLOYMENT_ARGS += -isysroot $(OSX_SDK)
IOS_DEPLOYMENT_ARGS += -isysroot $(IOS_SDK)
IOS6_DEPLOYMENT_ARGS += -isysroot $(IOS_SDK)
IOSSIM_DEPLOYMENT_ARGS += -isysroot $(IOSSIM_SDK)

CFLAGS.eprintf		:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.10.4		:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)

SANITIZER_MACOSX_DEPLOYMENT_ARGS := -mmacosx-version-min=10.7
SANITIZER_IOSSIM_DEPLOYMENT_ARGS := -mios-simulator-version-min=7.0 \
  -isysroot $(IOSSIM_SDK)
SANITIZER_CFLAGS := -fno-builtin -gline-tables-only -stdlib=libc++


CFLAGS.ios.i386		:= $(CFLAGS) $(IOSSIM_DEPLOYMENT_ARGS)
CFLAGS.ios.x86_64	:= $(CFLAGS) $(IOSSIM_DEPLOYMENT_ARGS)
CFLAGS.ios.armv7	:= $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.ios.armv7k	:= $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.ios.armv7s	:= $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.ios.arm64	:= $(CFLAGS) $(IOS6_DEPLOYMENT_ARGS)
CFLAGS.osx.i386		:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.osx.x86_64	:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.osx.x86_64h	:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.cc_kext.i386	:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.cc_kext.x86_64	:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.cc_kext.x86_64h	:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.cc_kext_ios.armv7	:= $(CFLAGS) $(IOS6_DEPLOYMENT_ARGS)
CFLAGS.cc_kext_ios.armv7k	:= $(CFLAGS) $(IOS6_DEPLOYMENT_ARGS)
CFLAGS.cc_kext_ios.armv7s	:= $(CFLAGS) $(IOS6_DEPLOYMENT_ARGS)
CFLAGS.cc_kext_ios.arm64	:= $(CFLAGS) $(IOS6_DEPLOYMENT_ARGS)

SANITIZER_LDFLAGS := -stdlib=libc++ -lc++ -lc++abi

ATOMIC_FUNCTIONS := \
	atomic_flag_clear \
	atomic_flag_clear_explicit \
	atomic_flag_test_and_set \
	atomic_flag_test_and_set_explicit \
	atomic_signal_fence \
	atomic_thread_fence

FP16_FUNCTIONS := \
	extendhfsf2 \
	truncdfhf2 \
	truncsfhf2

FUNCTIONS.eprintf := eprintf
FUNCTIONS.10.4 := eprintf floatundidf floatundisf floatundixf

FUNCTIONS.ios	    := divmodsi4 udivmodsi4 mulosi4 mulodi4 muloti4 \
                       $(ATOMIC_FUNCTIONS) $(FP16_FUNCTIONS)
# On x86, the divmod functions reference divsi.
FUNCTIONS.ios.i386    := $(FUNCTIONS.ios) \
                         divsi3 udivsi3
FUNCTIONS.ios.x86_64  := $(FUNCTIONS.ios.i386)
FUNCTIONS.ios.arm64   := mulsc3 muldc3 divsc3 divdc3 udivti3 umodti3 \
                         $(ATOMIC_FUNCTIONS)

FUNCTIONS.osx	:= mulosi4 mulodi4 muloti4 $(ATOMIC_FUNCTIONS) $(FP16_FUNCTIONS)

CCKEXT_PROFILE_FUNCTIONS := \
	InstrProfiling \
	InstrProfilingBuffer \
	InstrProfilingPlatformDarwin

CCKEXT_COMMON_FUNCTIONS := \
	$(CCKEXT_PROFILE_FUNCTIONS) \
	absvdi2 \
	absvsi2 \
	addvdi3 \
	addvsi3 \
	ashldi3 \
	ashrdi3 \
	bswapdi2 \
	bswapsi2 \
	clzdi2 \
	clzsi2 \
	cmpdi2 \
	ctzdi2 \
	ctzsi2 \
	divdc3 \
	divdi3 \
	divsc3 \
	divmodsi4 \
	udivmodsi4 \
	do_global_dtors \
	eprintf \
	extendhfsf2 \
	ffsdi2 \
	fixdfdi \
	fixsfdi \
	fixunsdfdi \
	fixunsdfsi \
	fixunssfdi \
	fixunssfsi \
	floatdidf \
	floatdisf \
	floatundidf \
	floatundisf \
	gcc_bcmp \
	lshrdi3 \
	moddi3 \
	muldc3 \
	muldi3 \
	mulsc3 \
	mulvdi3 \
	mulvsi3 \
	negdi2 \
	negvdi2 \
	negvsi2 \
	paritydi2 \
	paritysi2 \
	popcountdi2 \
	popcountsi2 \
	powidf2 \
	powisf2 \
	subvdi3 \
	subvsi3 \
	truncdfhf2 \
	truncsfhf2 \
	ucmpdi2 \
	udiv_w_sdiv \
	udivdi3 \
	udivmoddi4 \
	umoddi3

CCKEXT_ARM_FUNCTIONS := $(CCKEXT_COMMON_FUNCTIONS) \
	adddf3 \
	addsf3 \
	aeabi_cdcmpeq \
	aeabi_cdrcmple \
	aeabi_cfcmpeq \
	aeabi_cfrcmple \
	aeabi_dcmpeq \
	aeabi_dcmpge \
	aeabi_dcmpgt \
	aeabi_dcmple \
	aeabi_dcmplt \
	aeabi_drsub \
	aeabi_fcmpeq \
	aeabi_fcmpge \
	aeabi_fcmpgt \
	aeabi_fcmple \
	aeabi_fcmplt \
	aeabi_frsub \
	aeabi_idivmod \
	aeabi_uidivmod \
	cmpdf2 \
	cmpsf2 \
	div0 \
	divdf3 \
	divsf3 \
	divsi3 \
	extendsfdf2 \
	ffssi2 \
	fixdfsi \
	fixsfsi \
	floatsidf \
	floatsisf \
	floatunsidf \
	floatunsisf \
	comparedf2 \
	comparesf2 \
	modsi3 \
	muldf3 \
	mulsf3 \
	mulodi4 \
	negdf2 \
	negsf2 \
	subdf3 \
	subsf3 \
	switch16 \
	switch32 \
	switch8 \
	switchu8 \
	truncdfsf2 \
	udivsi3 \
	umodsi3 \
	unorddf2 \
	unordsf2

CCKEXT_ARMVFP_FUNCTIONS := $(CCKEXT_ARM_FUNCTIONS) \
	adddf3vfp \
	addsf3vfp \
	divdf3vfp \
	divsf3vfp \
	eqdf2vfp \
	eqsf2vfp \
	extendsfdf2vfp \
	fixdfsivfp \
	fixsfsivfp \
	fixunsdfsivfp \
	fixunssfsivfp \
	floatsidfvfp \
	floatsisfvfp \
	floatunssidfvfp \
	floatunssisfvfp \
	gedf2vfp \
	gesf2vfp \
	gtdf2vfp \
	gtsf2vfp \
	ledf2vfp \
	lesf2vfp \
	ltdf2vfp \
	ltsf2vfp \
	muldf3vfp \
	mulsf3vfp \
	nedf2vfp \
	nesf2vfp \
	subdf3vfp \
	subsf3vfp \
	truncdfsf2vfp \
	unorddf2vfp \
	unordsf2vfp

CCKEXT_ARM64_FUNCTIONS := \
	$(CCKEXT_PROFILE_FUNCTIONS) \
	divdc3 \
	divsc3 \
	muldc3 \
	mulsc3 \
	udivti3 \
	umodti3

FUNCTIONS.cc_kext_ios.armv7 := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext_ios.armv7k := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext_ios.armv7s := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext_ios.arm64 := $(CCKEXT_ARM64_FUNCTIONS)

CCKEXT_X86_FUNCTIONS := $(CCKEXT_COMMON_FUNCTIONS) \
	divxc3 \
	fixunsxfdi \
	fixunsxfsi \
	fixxfdi \
	floatdixf \
	floatundixf \
	mulxc3 \
	powixf2

FUNCTIONS.cc_kext.i386 := $(CCKEXT_X86_FUNCTIONS) \
	ffssi2 \
	i686.get_pc_thunk.eax \
	i686.get_pc_thunk.ebp \
	i686.get_pc_thunk.ebx \
	i686.get_pc_thunk.ecx \
	i686.get_pc_thunk.edi \
	i686.get_pc_thunk.edx \
	i686.get_pc_thunk.esi

FUNCTIONS.cc_kext.x86_64 := $(CCKEXT_X86_FUNCTIONS) \
	absvti2 \
	addvti3 \
	ashlti3 \
	ashrti3 \
	clzti2 \
	cmpti2 \
	ctzti2 \
	divti3 \
	ffsti2 \
	fixdfti \
	fixsfti \
	fixunsdfti \
	fixunssfti \
	fixunsxfti \
	fixxfti \
	floattidf \
	floattisf \
	floattixf \
	floatuntidf \
	floatuntisf \
	floatuntixf \
	lshrti3 \
	modti3 \
	multi3 \
	mulvti3 \
	negti2 \
	negvti2 \
	parityti2 \
	popcountti2 \
	subvti3 \
	ucmpti2 \
	udivmodti4 \
	udivti3 \
	umodti3

FUNCTIONS.cc_kext.x86_64h := $(FUNCTIONS.cc_kext.x86_64)

# FIXME: Currently, compiler-rt is missing implementations for a number of the
# functions that need to go into libcc_kext.a. Filter them out for now.
CCKEXT_MISSING_FUNCTIONS := \
	cmpdf2 cmpsf2 div0 \
	ffssi2 \
	udiv_w_sdiv unorddf2 unordsf2 bswapdi2 \
	bswapsi2 \
	gcc_bcmp \
	do_global_dtors \
	i686.get_pc_thunk.eax i686.get_pc_thunk.ebp i686.get_pc_thunk.ebx \
	i686.get_pc_thunk.ecx i686.get_pc_thunk.edi i686.get_pc_thunk.edx \
	i686.get_pc_thunk.esi \
	aeabi_cdcmpeq aeabi_cdrcmple aeabi_cfcmpeq aeabi_cfrcmple aeabi_dcmpeq \
	aeabi_dcmpge aeabi_dcmpgt aeabi_dcmple aeabi_dcmplt aeabi_drsub aeabi_fcmpeq \
	aeabi_fcmpge aeabi_fcmpgt aeabi_fcmple aeabi_fcmplt aeabi_frsub aeabi_idivmod \
	aeabi_uidivmod

FUNCTIONS.cc_kext_ios.armv7 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext_ios.armv7))
FUNCTIONS.cc_kext_ios.armv7k := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext_ios.armv7k))
FUNCTIONS.cc_kext_ios.armv7s := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext_ios.armv7s))
FUNCTIONS.cc_kext_ios.arm64 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext_ios.arm64))
FUNCTIONS.cc_kext.i386 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.i386))
FUNCTIONS.cc_kext.x86_64 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.x86_64))
FUNCTIONS.cc_kext.x86_64h := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.x86_64h))

KERNEL_USE.cc_kext := 1
KERNEL_USE.cc_kext_ios := 1

VISIBILITY_HIDDEN := 1

SHARED_LIBRARY_SUFFIX := dylib
