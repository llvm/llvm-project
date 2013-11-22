# These are the functions which clang needs when it is targetting a previous
# version of the OS. The issue is that the backend may use functions which were
# not present in the libgcc that shipped on the platform. In such cases, we link
# with a version of the library which contains private_extern definitions of all
# the extra functions which might be referenced.

Description := Static runtime libraries for clang/Darwin.

# A function that ensures we don't try to build for architectures that we
# don't have working toolchains for.
CheckArches = \
  $(shell \
    result=""; \
    for arch in $(1); do \
      if $(CC) -arch $$arch -c \
	  -integrated-as \
	  $(ProjSrcRoot)/make/platform/clang_darwin_test_input.c \
	  -isysroot $(ProjSrcRoot)/SDKs/darwin \
	  -o /dev/null > /dev/null 2> /dev/null; then \
        result="$$result$$arch "; \
      else \
	printf 1>&2 \
	  "warning: clang_darwin.mk: dropping arch '$$arch' from lib '$(2)'\n"; \
      fi; \
    done; \
    echo $$result)

XCRun = \
  $(shell \
    result=`xcrun -find $(1) 2> /dev/null`; \
    if [ "$$?" != "0" ]; then result=$(1); fi; \
    echo $$result)
XCRunSdkPath = \
  $(shell \
    result=`xcrun --sdk $(1) --show-sdk-path 2> /dev/null`; \
    if [ "$$?" != "0" ]; then result=""; fi; \
    echo $$result)
###

CC       := $(call XCRun,clang)
AR       := $(call XCRun,ar)
RANLIB   := $(call XCRun,ranlib)
STRIP    := $(call XCRun,strip)
LIPO     := $(call XCRun,lipo)
DSYMUTIL := $(call XCRun,dsymutil)

Configs :=
UniversalArchs :=

# Configuration solely for providing access to an eprintf symbol, which may
# still be referenced from Darwin system headers. This symbol is only ever
# needed on i386.
Configs += eprintf
UniversalArchs.eprintf := $(call CheckArches,i386,eprintf)

# Configuration for targetting 10.4. We need a few functions missing from
# libgcc_s.10.4.dylib. We only build x86 slices since clang doesn't really
# support targetting PowerPC.
Configs += 10.4
UniversalArchs.10.4 := $(call CheckArches,i386 x86_64,10.4)

# Configuration for targetting iOS for a couple of functions that didn't
# make it into libSystem.
Configs += ios
UniversalArchs.ios := $(call CheckArches,i386 x86_64 armv7,ios)

# Configuration for targetting OSX. These functions may not be in libSystem
# so we should provide our own.
Configs += osx
UniversalArchs.osx := $(call CheckArches,i386 x86_64,osx)

# Configuration for use with kernel/kexts.
Configs += cc_kext
UniversalArchs.cc_kext := $(call CheckArches,armv7 i386 x86_64,cc_kext)

# Configuration for use with kernel/kexts for iOS 5.0 and earlier (which used 
# a different code generation strategy).
Configs += cc_kext_ios5
UniversalArchs.cc_kext_ios5 := $(call CheckArches,x86_64 armv7,cc_kext_ios5)

# Configurations which define the profiling support functions.
Configs += profile_osx
UniversalArchs.profile_osx := $(call CheckArches,i386 x86_64,profile_osx)
Configs += profile_ios
UniversalArchs.profile_ios := $(call CheckArches,i386 x86_64 armv7,profile_ios)

# Configurations which define the ASAN support functions.
Configs += asan_osx_dynamic
UniversalArchs.asan_osx_dynamic := $(call CheckArches,i386 x86_64,asan_osx_dynamic)

IOSSIM_SDK_PATH := $(call XCRunSdkPath,iphonesimulator)
ifneq ($(IOSSIM_SDK_PATH),)
Configs += asan_iossim_dynamic
UniversalArchs.asan_iossim_dynamic := $(call CheckArches,i386 x86_64,asan_iossim_dynamic)
endif

Configs += ubsan_osx
UniversalArchs.ubsan_osx := $(call CheckArches,i386 x86_64,ubsan_osx)

# Darwin 10.6 has a bug in cctools that makes it unable to use ranlib on our ARM
# object files. If we are on that platform, strip out all ARM archs. We still
# build the libraries themselves so that Clang can find them where it expects
# them, even though they might not have an expected slice.
ifneq ($(shell test -x /usr/bin/sw_vers && sw_vers -productVersion | grep 10.6),)
UniversalArchs.ios := $(filter-out armv7, $(UniversalArchs.ios))
UniversalArchs.cc_kext := $(filter-out armv7, $(UniversalArchs.cc_kext))
UniversalArchs.cc_kext_ios5 := $(filter-out armv7, $(UniversalArchs.cc_kext_ios5))
UniversalArchs.profile_ios := $(filter-out armv7, $(UniversalArchs.profile_ios))
endif

# If RC_SUPPORTED_ARCHS is defined, treat it as a list of the architectures we
# are intended to support and limit what we try to build to that.
#
# We make sure to remove empty configs if we end up dropping all the requested
# archs for a particular config.
ifneq ($(RC_SUPPORTED_ARCHS),)
$(foreach config,$(Configs),\
  $(call Set,UniversalArchs.$(config),\
	$(filter $(RC_SUPPORTED_ARCHS),$(UniversalArchs.$(config))))\
  $(if $(UniversalArchs.$(config)),,\
	$(call Set,Configs,$(filter-out $(config),$(Configs)))))
endif

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

# Use our stub SDK as the sysroot to support more portable building.
OSX_DEPLOYMENT_ARGS += -isysroot $(ProjSrcRoot)/SDKs/darwin
IOS_DEPLOYMENT_ARGS += -isysroot $(ProjSrcRoot)/SDKs/darwin
IOS6_DEPLOYMENT_ARGS += -isysroot $(ProjSrcRoot)/SDKs/darwin
IOSSIM_DEPLOYMENT_ARGS += -isysroot $(ProjSrcRoot)/SDKs/darwin

CFLAGS.eprintf		:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.10.4		:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
# FIXME: We can't build ASAN with our stub SDK yet.
CFLAGS.asan_osx_dynamic := \
	$(CFLAGS) -mmacosx-version-min=10.6 -fno-builtin \
	-gline-tables-only \
	-DMAC_INTERPOSE_FUNCTIONS=1 \
  -DASAN_FLEXIBLE_MAPPING_AND_OFFSET=1

CFLAGS.asan_iossim_dynamic := \
	$(CFLAGS) -mios-simulator-version-min=7.0 \
        -isysroot $(IOSSIM_SDK_PATH) \
        -fno-builtin \
	-gline-tables-only \
	-DMAC_INTERPOSE_FUNCTIONS=1 \
  -DASAN_FLEXIBLE_MAPPING_AND_OFFSET=1

CFLAGS.ubsan_osx	:= $(CFLAGS) -mmacosx-version-min=10.6 -fno-builtin

CFLAGS.ios.i386		:= $(CFLAGS) $(IOSSIM_DEPLOYMENT_ARGS)
CFLAGS.ios.x86_64	:= $(CFLAGS) $(IOSSIM_DEPLOYMENT_ARGS)
CFLAGS.ios.armv7	:= $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.ios.armv7f	:= $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.ios.armv7k	:= $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.ios.armv7s	:= $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.osx.i386		:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.osx.x86_64	:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.cc_kext.i386	:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.cc_kext.x86_64	:= $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.cc_kext.armv7	:= $(CFLAGS) $(IOS6_DEPLOYMENT_ARGS)
CFLAGS.cc_kext.armv7f	:= $(CFLAGS) $(IOS6_DEPLOYMENT_ARGS)
CFLAGS.cc_kext.armv7k	:= $(CFLAGS) $(IOS6_DEPLOYMENT_ARGS)
CFLAGS.cc_kext.armv7s	:= $(CFLAGS) $(IOS6_DEPLOYMENT_ARGS)
CFLAGS.cc_kext_ios5.armv7  := $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.cc_kext_ios5.armv7f := $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.cc_kext_ios5.armv7k := $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.cc_kext_ios5.armv7s := $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.profile_osx.i386   := $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.profile_osx.x86_64 := $(CFLAGS) $(OSX_DEPLOYMENT_ARGS)
CFLAGS.profile_ios.i386   := $(CFLAGS) $(IOSSIM_DEPLOYMENT_ARGS)
CFLAGS.profile_ios.x86_64 := $(CFLAGS) $(IOSSIM_DEPLOYMENT_ARGS)
CFLAGS.profile_ios.armv7  := $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.profile_ios.armv7f := $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.profile_ios.armv7k := $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)
CFLAGS.profile_ios.armv7s := $(CFLAGS) $(IOS_DEPLOYMENT_ARGS)

# Configure the asan_osx_dynamic library to be built shared.
SHARED_LIBRARY.asan_osx_dynamic := 1
LDFLAGS.asan_osx_dynamic := -lstdc++ -undefined dynamic_lookup

# Configure the asan_iossim_dynamic library to be built shared.
SHARED_LIBRARY.asan_iossim_dynamic := 1
# configure+make uses Clang, so we're using isysroot instead of --sysroot
# or -Wl,-syslibroot.
LDFLAGS.asan_iossim_dynamic := -undefined dynamic_lookup \
  -Wl,-ios_simulator_version_min,7.0.0 \
  -mios-simulator-version-min=7.0 -isysroot $(IOSSIM_SDK_PATH)

FUNCTIONS.eprintf := eprintf
FUNCTIONS.10.4 := eprintf floatundidf floatundisf floatundixf

FUNCTIONS.ios	    := divmodsi4 udivmodsi4 mulosi4 mulodi4 muloti4
# On x86, the divmod functions reference divsi.
FUNCTIONS.ios.i386   := $(FUNCTIONS.ios) \
                        divsi3 udivsi3
FUNCTIONS.ios.x86_64 := $(FUNCTIONS.ios) \
                        divsi3 udivsi3

FUNCTIONS.osx	:= mulosi4 mulodi4 muloti4

FUNCTIONS.profile_osx := GCDAProfiling
FUNCTIONS.profile_ios := GCDAProfiling

FUNCTIONS.asan_osx_dynamic := $(AsanFunctions) $(InterceptionFunctions) \
                              $(SanitizerCommonFunctions) \
	                      $(AsanDynamicFunctions)

FUNCTIONS.asan_iossim_dynamic := $(AsanFunctions) $(InterceptionFunctions) \
                                 $(SanitizerCommonFunctions) \
	                         $(AsanDynamicFunctions)

FUNCTIONS.ubsan_osx := $(UbsanFunctions) $(UbsanCXXFunctions) \
                       $(SanitizerCommonFunctions)

CCKEXT_COMMON_FUNCTIONS := \
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

FUNCTIONS.cc_kext.armv7 := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext.armv7f := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext.armv7k := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext.armv7s := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext_ios5.armv7 := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext_ios5.armv7f := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext_ios5.armv7k := $(CCKEXT_ARMVFP_FUNCTIONS)
FUNCTIONS.cc_kext_ios5.armv7s := $(CCKEXT_ARMVFP_FUNCTIONS)

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

FUNCTIONS.cc_kext.armv7 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.armv7))
FUNCTIONS.cc_kext.armv7f := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.armv7f))
FUNCTIONS.cc_kext.armv7k := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.armv7k))
FUNCTIONS.cc_kext.armv7s := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.armv7s))
FUNCTIONS.cc_kext_ios5.armv7 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext_ios5.armv7))
FUNCTIONS.cc_kext_ios5.armv7f := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext_ios5.armv7f))
FUNCTIONS.cc_kext_ios5.armv7k := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext_ios5.armv7k))
FUNCTIONS.cc_kext_ios5.armv7s := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext_ios5.armv7s))
FUNCTIONS.cc_kext.i386 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.i386))
FUNCTIONS.cc_kext.x86_64 := \
	$(filter-out $(CCKEXT_MISSING_FUNCTIONS),$(FUNCTIONS.cc_kext.x86_64))

KERNEL_USE.cc_kext := 1
KERNEL_USE.cc_kext_ios5 := 1

VISIBILITY_HIDDEN := 1

SHARED_LIBRARY_SUFFIX := dylib
