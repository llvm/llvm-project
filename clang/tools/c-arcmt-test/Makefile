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
	   clangARCMigrate.a \
	   clangIndex.a \
	   clangFormat.a \
	   clangTooling.a \
	   clangToolingCore.a \
	   clangRewriteFrontend.a \
	   clangRewrite.a \
	   clangFrontend.a clangDriver.a \
	   clangStaticAnalyzerCheckers.a clangStaticAnalyzerCore.a \
	   clangSerialization.a clangParse.a clangSema.a \
	   clangAnalysis.a clangEdit.a clangAST.a clangLex.a clangAPINotes.a \
           clangBasic.a

include $(CLANG_LEVEL)/Makefile
