##===- clang/runtime/compiler-rt/Makefile ------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
#
# This file defines support for building the Clang runtime libraries (which are
# implemented by compiler-rt) and placing them in the proper locations in the
# Clang resources directory (i.e., where the driver expects them).
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..
include $(CLANG_LEVEL)/Makefile

CLANG_VERSION := $(word 3,$(shell grep "CLANG_VERSION " \
	$(PROJ_OBJ_DIR)/$(CLANG_LEVEL)/include/clang/Basic/Version.inc))

ResourceDir := $(PROJ_OBJ_ROOT)/$(BuildMode)/lib/clang/$(CLANG_VERSION)
PROJ_resources := $(DESTDIR)$(PROJ_prefix)/lib/clang/$(CLANG_VERSION)

ResourceLibDir := $(ResourceDir)/lib
PROJ_resources_lib := $(PROJ_resources)/lib

# Expect compiler-rt to be in llvm/projects/compiler-rt
COMPILERRT_SRC_ROOT := $(LLVM_SRC_ROOT)/projects/compiler-rt

# We don't currently support building runtime libraries when we are
# cross-compiling. The issue is that we really want to be set up so that the
# available compiler targets are independent of the current build.
#
# Since we have to build the runtime libraries for the target, it requires we
# have a cross compiler from the build machine to the target. Although in the
# case where for the current build (host == target), we do have such a cross
# compiler, but not defined in a way that is easy for us to reuse. Regardless,
# that also wouldn't help for other possible compiler configurations.
#
# Thus, the simple set up we currently use is to assume that we will be using
# the just built Clang to compile the compiler-rt libraries. As we grow better
# cross compilation support inside Clang and tool support in LLVM, this makes it
# easier for us to achieve the goal of having the compiler targets be easily
# selected at configure time. However, this design does currently preclude the
# building of compiler-rt libraries when the Clang itself is being cross
# compiled.
#
# There are three possible solutions:
#  1. Require building a build-target version of Clang when cross compiling. This
#     is simplest, but als greatly increases the build time of cross builds.
#
#  2. Require cross builds have a build-target version of Clang available for
#     use. This is a reasonable compromise on #1, as the compiler-rt libraries
#     are simple enough that there is not a strong desire to ensure they are
#     built with the exact version of Clang being used. Similarly, as Clang
#     becomes a better cross compiler it is also increasingly more likely that
#     the cross compiler being used will already be a version of Clang.
#
#  3. Come up with an alternate mechanism to define all the toolchain
#     information that compiler-rt would need to build libraries for all the
#     requested targets. This might be a simple short term solution, but is
#     likely to be unwieldly and irritating to maintain in the long term.
ifneq ($(LLVM_CROSS_COMPILING),1)
ifneq ($(CLANG_NO_RUNTIME),1)
ifeq ($(shell test -d $(COMPILERRT_SRC_ROOT) && echo OK),OK)

# Select the compiler-rt configuration to use, and install directory.
#
# FIXME: Eventually, we want some kind of configure support for this. We want to
# build/install runtime libraries for as many targets as clang was configured to
# support.
RuntimeDirs :=
ifeq ($(OS),Darwin)
RuntimeDirs += darwin
RuntimeLibrary.darwin.Configs := \
	eprintf.a 10.4.a osx.a ios.a cc_kext.a cc_kext_ios5.a \
	asan_osx.a asan_osx_dynamic.dylib \
	profile_osx.a profile_ios.a
endif

# On Linux, include a library which has all the runtime functions.
ifeq ($(OS),Linux)
RuntimeDirs += linux
RuntimeLibrary.linux.Configs :=

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

# We currently only try to generate runtime libraries on x86.
ifeq ($(ARCH),x86)
RuntimeLibrary.linux.Configs += \
	full-i386.a profile-i386.a asan-i386.a
endif

ifeq ($(ARCH),x86_64)
RuntimeLibrary.linux.Configs += \
	full-x86_64.a profile-x86_64.a asan-x86_64.a tsan-x86_64.a
# We need to build 32-bit ASan library on 64-bit platform, and add it to the
# list of runtime libraries to make "clang -faddress-sanitizer -m32" work.
# We check that Clang can produce working 32-bit binaries by compiling a simple
# executable.
test_source = $(LLVM_SRC_ROOT)/tools/clang/runtime/compiler-rt/clang_linux_test_input.c
ifeq ($(call TryCompile,$(ToolDir)/clang,$(test_source),-m32),0)
RuntimeLibrary.linux.Configs += asan-i386.a
endif
ifneq ($(LLVM_ANDROID_TOOLCHAIN_DIR),)
RuntimeLibrary.linux.Configs += asan-arm-android.so
endif
endif

endif

####
# The build rules below are designed to be generic and should only need to be
# modified based on changes in the compiler-rt layout or build system.
####

# Rule to build the compiler-rt libraries we need.
#
# We build all the libraries in a single shot to avoid recursive make as much as
# possible.
BuildRuntimeLibraries:
	$(Verb) $(MAKE) -C $(COMPILERRT_SRC_ROOT) \
	  ProjSrcRoot=$(COMPILERRT_SRC_ROOT) \
	  ProjObjRoot=$(PROJ_OBJ_DIR) \
	  CC="$(ToolDir)/clang" \
	  LLVM_ANDROID_TOOLCHAIN_DIR="$(LLVM_ANDROID_TOOLCHAIN_DIR)" \
	  $(RuntimeDirs:%=clang_%)
.PHONY: BuildRuntimeLibraries
CleanRuntimeLibraries:
	$(Verb) $(MAKE) -C $(COMPILERRT_SRC_ROOT) \
	  ProjSrcRoot=$(COMPILERRT_SRC_ROOT) \
	  ProjObjRoot=$(PROJ_OBJ_DIR) \
	  clean
.PHONY: CleanRuntimeLibraries

$(PROJ_resources_lib):
	$(Verb) $(MKDIR) $@

# Expand rules for copying/installing each individual library. We can't use
# implicit rules here because we need to match against multiple things.
define RuntimeLibraryTemplate
$(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.a: BuildRuntimeLibraries
	@true
$(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.so: BuildRuntimeLibraries
	@true
$(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.dylib: BuildRuntimeLibraries
	@true
.PRECIOUS: $(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.a

# Rule to copy the libraries to their resource directory location.
$(ResourceLibDir)/$1/libclang_rt.%.a: \
		$(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.a \
		$(ResourceLibDir)/$1/.dir
	$(Echo) Copying runtime library $1/$$* to build dir
	$(Verb) cp $(PROJ_OBJ_DIR)/clang_$1/$$*/libcompiler_rt.a $$@
$(ResourceLibDir)/$1/libclang_rt.%.so: \
		$(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.so \
		$(ResourceLibDir)/$1/.dir
	$(Echo) Copying runtime library $1/$$* to build dir
	$(Verb) cp $(PROJ_OBJ_DIR)/clang_$1/$$*/libcompiler_rt.so $$@
$(ResourceLibDir)/$1/libclang_rt.%.dylib: \
		$(PROJ_OBJ_DIR)/clang_$1/%/libcompiler_rt.dylib \
		$(ResourceLibDir)/$1/.dir
	$(Echo) Copying runtime library $1/$$* to build dir
	$(Verb) cp $(PROJ_OBJ_DIR)/clang_$1/$$*/libcompiler_rt.dylib $$@
	$(Echo) Fixing LC_ID_DYLIB of $$@
	$(Verb) install_name_tool $$@ -id $$@
RuntimeLibrary.$1: \
		$(RuntimeLibrary.$1.Configs:%=$(ResourceLibDir)/$1/libclang_rt.%)
.PHONY: RuntimeLibrary.$1

$(PROJ_resources_lib)/$1: $(PROJ_resources_lib)
	$(Verb) $(MKDIR) $$@

$(PROJ_resources_lib)/$1/libclang_rt.%.a: \
		$(ResourceLibDir)/$1/libclang_rt.%.a | $(PROJ_resources_lib)/$1
	$(Echo) Installing compiler runtime library: $1/$$*
	$(Verb) $(DataInstall) $$< $(PROJ_resources_lib)/$1
$(PROJ_resources_lib)/$1/libclang_rt.%.so: \
		$(ResourceLibDir)/$1/libclang_rt.%.so | $(PROJ_resources_lib)/$1
	$(Echo) Installing compiler runtime library: $1/$$*
	$(Verb) $(DataInstall) $$< $(PROJ_resources_lib)/$1
$(PROJ_resources_lib)/$1/libclang_rt.%.dylib: \
		$(ResourceLibDir)/$1/libclang_rt.%.dylib | $(PROJ_resources_lib)/$1
	$(Echo) Installing compiler runtime library: $1/$$*
	$(Verb) $(DataInstall) $$< $(PROJ_resources_lib)/$1

# Rule to install runtime libraries.
RuntimeLibraryInstall.$1: \
		$(RuntimeLibrary.$1.Configs:%=$(PROJ_resources_lib)/$1/libclang_rt.%)
.PHONY: RuntimeLibraryInstall.$1
endef
$(foreach lib,$(RuntimeDirs), $(eval $(call RuntimeLibraryTemplate,$(lib))))

# Hook into the standard Makefile rules.
all-local:: $(RuntimeDirs:%=RuntimeLibrary.%)
install-local:: $(RuntimeDirs:%=RuntimeLibraryInstall.%)
clean-local:: CleanRuntimeLibraries

endif
endif
endif
