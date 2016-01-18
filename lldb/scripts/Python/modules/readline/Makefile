##===- scripts/Python/modules/readline/Makefile ------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

# Skip this entire Makefile if python is disabled.
ifeq (,$(findstring -DLLDB_DISABLE_PYTHON,$(CXXFLAGS)))

LEVEL := ../../../../../..
LLDB_LEVEL := ../../../..

LIBRARYNAME = readline

NO_BUILD_ARCHIVE = 1
LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1
LOADABLE_MODULE = 1

PYTHON_INC_DIR = $(shell python-config --includes)

# Include all archives in the shared lib
USEDLIBS :=

include $(LLDB_LEVEL)/../../Makefile.config

LINK_COMPONENTS :=

include $(LEVEL)/Makefile.common

# include python headers
CPP.Flags += $(PYTHON_INC_DIR)

ifeq ($(HOST_OS),Darwin)
    LLVMLibsOptions += -Wl,-all_load
    # set dylib internal version number to llvmCore submission number
    ifdef LLDB_SUBMIT_VERSION
        LLVMLibsOptions += -Wl,-current_version \
                           -Wl,$(LLDB_SUBMIT_VERSION).$(LLDB_SUBMIT_SUBVERSION) \
                           -Wl,-compatibility_version -Wl,1
    endif
    # extra options to override libtool defaults
    LVMLibsOptions += -F/System/Library/Frameworks -F/System/Library/PrivateFrameworks
    LLVMLibsOptions += -framework Foundation -framework CoreFoundation
    LLVMLibsOptions += -framework CoreServices -framework Carbon -framework Security
    LLVMLibsOptions += -framework DebugSymbols $(PYTHON_BUILD_FLAGS) -lobjc
    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on command line
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions  += -Wl,-install_name \
                           -Wl,"@executable_path/../lib/$(LIBRARYNAME)$(SHLIBEXT)"
    endif
endif

ifeq ($(HOST_OS), $(filter $(HOST_OS), Linux GNU GNU/kFreeBSD))
    # Include everything from the .a's into the shared library.
    ProjLibsOptions := -Wl,--whole-archive $(ProjLibsOptions) \
                       -Wl,--no-whole-archive
    # Link in libedit
    # LLVMLibsOptions += -ledit
    LLVMLibsOptions += -Wl,--soname,$(LIBRARYNAME)$(SHLIBEXT)
endif

ifeq ($(HOST_OS),FreeBSD)
    # Include everything from the .a's into the shared library.
    ProjLibsOptions := -Wl,--whole-archive $(ProjLibsOptions) \
                       -Wl,--no-whole-archive
    # Allow unresolved symbols.
    LLVMLibsOptions += -Wl,--allow-shlib-undefined
    # Link in libedit
    # LLVMLibsOptions += -L/usr/local/lib -ledit
endif

# FIXME: dynamically construct the version from `python -V`
PYTHON_VERSION:=2.7
LLDB_PYTHON_MODULE_REL_DIR:=python$(PYTHON_VERSION)/site-packages
LLDB_PYTHON_MODULE_DIR:=$(LibDir)/$(LLDB_PYTHON_MODULE_REL_DIR)

# Target to move readline module from shared lib build location to
# local python module directory.
all-local:: $(LLDB_PYTHON_MODULE_DIR)/$(LIBRARYNAME)$(SHLIBEXT)

$(LLDB_PYTHON_MODULE_DIR)/$(LIBRARYNAME)$(SHLIBEXT): $(SharedLibDir)/$(LIBRARYNAME)$(SHLIBEXT)
	$(Echo) Staging $(BuildMode) $(LIBRARYNAME)$(SHLIBEXT) to $(LLDB_PYTHON_MODULE_DIR)
	$(Verb) $(MKDIR) "$(LLDB_PYTHON_MODULE_DIR)"
	$(Verb) $(ProgInstall) $(SharedLibDir)/$(LIBRARYNAME)$(SHLIBEXT) $(LLDB_PYTHON_MODULE_DIR)

# Target to move the shared library from the build python lib dir to
# the install python lib dir.
install-local:: $(LLDB_PYTHON_MODULE_DIR)/$(LIBRARYNAME)$(SHLIBEXT)
	$(Echo) Installing $(BuildMode) $(LLDB_PYTHON_MODULE_DIR)/$(LIBRARYNAME)$(SHLIBEXT) to $(DESTDIR)$(prefix)/lib/$(LLDB_PYTHON_MODULE_REL_DIR)
	$(Verb) $(MKDIR) "$(DESTDIR)$(prefix)/lib/$(LLDB_PYTHON_MODULE_REL_DIR)"
	$(Verb) $(ProgInstall) "$(LLDB_PYTHON_MODULE_DIR)/$(LIBRARYNAME)$(SHLIBEXT)" "$(DESTDIR)$(prefix)/lib/$(LLDB_PYTHON_MODULE_REL_DIR)"
	$(Verb) $(RM) "$(DESTDIR)$(prefix)/lib/$(LIBRARYNAME)$(SHLIBEXT)"

endif  # if !defined(LLDB_DISABLE_PYTHON)
