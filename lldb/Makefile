##===- Makefile --------------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

# If LLDB_LEVEL is not set, then we are the top-level Makefile. Otherwise, we
# are being included from a subdirectory makefile.

ifndef LLDB_LEVEL


IS_TOP_LEVEL := 1
LLDB_LEVEL := .
DIRS := include scripts source lib tools

PARALLEL_DIRS :=
endif

###
# Common Makefile code, shared by all LLDB Makefiles.

# Set LLVM source root level.
LEVEL := $(LLDB_LEVEL)/../..

# Include LLVM common makefile.
include $(LEVEL)/Makefile.common

# Set common LLDB build flags.
CPP.Flags += -I$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/include
CPP.Flags += -I$(PROJ_OBJ_DIR)/$(LLDB_LEVEL)/include
CPP.Flags += -I$(LLVM_SRC_ROOT)/tools/clang/include
CPP.Flags += -I$(LLVM_OBJ_ROOT)/tools/clang/include
CPP.Flags += -I$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/source
CPP.Flags += -I$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/source/Utility
CPP.Flags += -I$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/source/Plugins/Process/Utility
CPP.Flags += -I$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/source/Plugins/Process/POSIX

# Disable python and curses on mingw build
ifeq ($(HOST_OS),MingW)
CXXFLAGS += -DLLDB_DISABLE_PYTHON -DLLDB_DISABLE_CURSES
endif

ifeq (,$(findstring -DLLDB_DISABLE_PYTHON,$(CXXFLAGS)))
# Set Python include directory
PYTHON_CONFIG?=	python-config
PYTHON_INC_DIR = $(shell $(PYTHON_CONFIG) --includes)
CPP.Flags +=   $(PYTHON_INC_DIR)
endif

ifeq ($(HOST_OS),Darwin)
CPP.Flags += $(subst -I,-I$(SDKROOT),$(PYTHON_INC_DIR))
CPP.Flags += -F$(SDKROOT)/System/Library/Frameworks
CPP.Flags += -F$(SDKROOT)/System/Library/PrivateFrameworks
CPP.Flags += -I$(SDKROOT)/usr/include/libxml2
endif
ifdef LLDB_VENDOR
CPP.Flags += -DLLDB_VENDOR='"$(LLDB_VENDOR) "'
endif

# If building on a 32-bit system, make sure off_t can store offsets > 2GB
ifneq "$(HOST_ARCH)" "x86_64"
CPP.Flags += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
endif

# Disable -fstrict-aliasing. Darwin disables it by default (and LLVM doesn't
# work with it enabled with GCC), Clang/llvm-gc don't support it yet, and newer
# GCC's have false positive warnings with it on Linux (which prove a pain to
# fix). For example:
#   http://gcc.gnu.org/PR41874
#   http://gcc.gnu.org/PR41838
#
# We can revisit this when LLVM/Clang support it.
CXX.Flags += -fno-strict-aliasing

# Do not warn about pragmas.  In particular, we are looking to ignore the
# "#pragma mark" construct which GCC warns about on platforms other than Darwin.
EXTRA_OPTIONS += -Wno-unknown-pragmas

# Drop -Wsign-compare, which we are not currently clean with.
EXTRA_OPTIONS += -Wno-sign-compare

###
# LLDB Top Level specific stuff.

ifeq ($(IS_TOP_LEVEL),1)

ifneq ($(PROJ_SRC_ROOT),$(PROJ_OBJ_ROOT))
$(RecursiveTargets)::
	$(Verb) if [ ! -f test/Makefile ]; then \
	  $(MKDIR) test; \
	  $(CP) $(PROJ_SRC_DIR)/test/Makefile test/Makefile; \
	fi
endif

test::
	@ $(MAKE) -C test

#report::
#	@ $(MAKE) -C test report

#clean::
#	@ $(MAKE) -C test clean

tags::
	$(Verb) etags `find . -type f -name '*.h' -or -name '*.cpp' | \
	  grep -v /lib/Headers | grep -v /test/`

cscope.files:
	find tools lib include -name '*.cpp' \
	                    -or -name '*.def' \
	                    -or -name '*.td' \
	                    -or -name '*.h' > cscope.files

.PHONY: test report clean cscope.files

endif
