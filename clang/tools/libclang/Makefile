##===- tools/libclang/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..
LIBRARYNAME = clang

EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/libclang.exports

LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1

LINK_COMPONENTS := support mc
USEDLIBS = clangARCMigrate.a clangRewrite.a clangFrontend.a clangDriver.a \
     clangSerialization.a \
		 clangParse.a clangSema.a clangEdit.a clangAnalysis.a \
		 clangAST.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile

# Add soname to the library.
ifeq ($(HOST_OS), $(filter $(HOST_OS), Linux FreeBSD GNU))
        LDFLAGS += -Wl,-soname,lib$(LIBRARYNAME)$(SHLIBEXT)
endif

##===----------------------------------------------------------------------===##
# FIXME: This is copied from the 'lto' makefile.  Should we share this?
##===----------------------------------------------------------------------===##

ifeq ($(HOST_OS),Darwin)
    LLVMLibsOptions += -Wl,-compatibility_version,1

    # Set dylib internal version number to submission number.
    ifdef LLVM_SUBMIT_VERSION
        LLVMLibsOptions += -Wl,-current_version \
                           -Wl,$(LLVM_SUBMIT_VERSION).$(LLVM_SUBMIT_SUBVERSION)
    endif

    # Extra options to override libtool defaults.
    LLVMLibsOptions += -Wl,-dead_strip -Wl,-seg1addr,0xE0000000 

    # Mac OS X 10.4 and earlier tools do not allow a second -install_name on command line
    DARWIN_VERS := $(shell echo $(TARGET_TRIPLE) | sed 's/.*darwin\([0-9]*\).*/\1/')
    ifneq ($(DARWIN_VERS),8)
       LLVMLibsOptions += -Wl,-install_name \
                          -Wl,"@rpath/lib$(LIBRARYNAME)$(SHLIBEXT)"
    endif
endif
