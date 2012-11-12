##===- examples/clang-interpreter/Makefile -----------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..

TOOLNAME = clang-interpreter
NO_INSTALL = 1

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

LINK_COMPONENTS := jit interpreter nativecodegen bitreader bitwriter ipo \
	linker selectiondag asmparser instrumentation
USEDLIBS = clangFrontend.a clangSerialization.a clangDriver.a clangCodeGen.a \
           clangParse.a clangSema.a clangStaticAnalyzerFrontend.a \
           clangStaticAnalyzerCheckers.a clangStaticAnalyzerCore.a \
           clangAnalysis.a clangRewriteCore.a clangRewriteFrontend.a \
           clangEdit.a clangAST.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile
