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

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser support mc
USEDLIBS = clang.a clangFrontend.a clangDriver.a \
	   clangTooling.a \
	   clangSerialization.a clangParse.a clangSema.a \
	   clangAnalysis.a clangEdit.a clangAST.a clangLex.a \
	   clangBasic.a

include $(CLANG_LEVEL)/Makefile

LIBS += $(LIBXML2_LIBS)
CPPFLAGS += $(LIBXML2_INC)
