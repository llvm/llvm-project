##===- unittests/Tooling/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../..
TESTNAME = Tooling
include $(CLANG_LEVEL)/../../Makefile.config
LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser support mc
USEDLIBS = clangTooling.a clangFrontend.a clangSerialization.a clangDriver.a \
           clangParse.a clangRewriteCore.a clangRewriteFrontend.a \
	   clangSema.a clangAnalysis.a clangEdit.a \
           clangAST.a clangASTMatchers.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/unittests/Makefile
