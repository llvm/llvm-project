##===- unittests/cpp11-migrate/Makefile --------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL = ../../../..
include $(CLANG_LEVEL)/../../Makefile.config

TESTNAME = Cpp11MigrateTests
LINK_COMPONENTS := asmparser bitreader support mc
USEDLIBS = clangTooling.a clangFrontend.a clangSerialization.a clangDriver.a \
		   clangRewriteFrontend.a clangRewriteCore.a clangParse.a \
		   clangSema.a clangAnalysis.a \
		   clangAST.a clangASTMatchers.a clangEdit.a clangLex.a clangBasic.a \
		   migrateCore.a

include $(CLANG_LEVEL)/Makefile
MAKEFILE_UNITTEST_NO_INCLUDE_COMMON := 1
CPP.Flags += -I$(PROJ_SRC_DIR)/../../cpp11-migrate
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest
