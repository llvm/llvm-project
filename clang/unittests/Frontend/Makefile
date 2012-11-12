##===- unittests/Frontend/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../..
TESTNAME = Frontend
include $(CLANG_LEVEL)/../../Makefile.config
LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser support mc
USEDLIBS = clangFrontendTool.a clangFrontend.a clangDriver.a \
           clangSerialization.a clangCodeGen.a clangParse.a clangSema.a \
           clangStaticAnalyzerCheckers.a clangStaticAnalyzerCore.a \
           clangARCMigrate.a clangRewriteCore.a \
		   clangRewriteFrontend.a clangEdit.a \
           clangAnalysis.a clangAST.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/unittests/Makefile
