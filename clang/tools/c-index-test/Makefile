##===- tools/index-test/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
CLANG_LEVEL := ../..

TOOLNAME = c-index-test

# If a separate install prefix was specified for internal tools, use it
# when installing c-index-test.
INTERNAL_TOOL = 1

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(CLANG_LEVEL)/../../Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) \
	AsmParser \
	BitReader \
	BitWriter \
	IPO \
	MC \
	ObjCARCOpts \
	Option \
	Support

# Note that 'USEDLIBS' must include all of the core clang libraries
# when -static is given to linker on cygming.
USEDLIBS = clang.a \
	   clangCodeGen.a \
	   clangIndex.a clangFormat.a clangRewrite.a \
	   clangFrontend.a clangDriver.a \
	   clangTooling.a \
	   clangToolingCore.a \
	   clangSerialization.a clangParse.a clangSema.a \
	   clangAnalysis.a clangEdit.a clangAST.a clangLex.a \
	   clangAPINotes.a clangBasic.a

include $(CLANG_LEVEL)/Makefile

LIBS += $(LIBXML2_LIBS)

# Headers in $(LIBXML2_INC) should not be checked with clang's -Wdocumentation.
# Use -isystem instead of -I then.
# FIXME: Could autoconf detect clang or availability of -isystem?
ifneq ($(findstring -Wdocumentation,$(OPTIMIZE_OPTION)),)
CPPFLAGS += $(subst -I,-isystem ,$(LIBXML2_INC))
else
CPPFLAGS += $(LIBXML2_INC)
endif
