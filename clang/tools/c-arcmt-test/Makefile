##===- tools/c-arcmt-test/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
CLANG_LEVEL := ../..

TOOLNAME = c-arcmt-test

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

# Don't install this. It is used for tests.
NO_INSTALL = 1

# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(CLANG_LEVEL)/../../Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser support mc

# Note that 'USEDLIBS' must include all of the core clang libraries
# as clang.dll is unavailable on cygming yet.
USEDLIBS = clang.a \
	   clangARCMigrate.a \
	   clangRewriteFrontend.a \
	   clangRewriteCore.a \
	   clangFrontend.a clangDriver.a \
	   clangSerialization.a clangParse.a clangSema.a \
	   clangAnalysis.a clangEdit.a clangAST.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile
