##===- unittests/libclang/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../..
TESTNAME = libclang
LINK_LIBS_IN_SHARED := 1

include $(CLANG_LEVEL)/../../Makefile.config
LINK_COMPONENTS := $(TARGETS_TO_BUILD) \
	AsmParser \
	BitReader \
	BitWriter \
	IPO \
	MC \
	ObjCArcOpts \
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

include $(CLANG_LEVEL)/unittests/Makefile
