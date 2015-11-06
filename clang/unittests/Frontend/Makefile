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
LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader support mc option
USEDLIBS = clangFrontendTool.a clangFrontend.a clangDriver.a \
           clangSerialization.a clangCodeGen.a clangParse.a clangSema.a \
           clangStaticAnalyzerCheckers.a clangStaticAnalyzerCore.a \
           clangARCMigrate.a clangRewrite.a \
		   clangRewriteFrontend.a clangEdit.a \
           clangAnalysis.a clangAST.a clangLex.a clangAPINotes.a clangBasic.a

include $(CLANG_LEVEL)/unittests/Makefile
