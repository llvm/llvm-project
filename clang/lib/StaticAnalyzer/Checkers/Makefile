##===- clang/lib/Checker/Makefile --------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# This implements analyses built on top of source-level CFGs. 
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../../..
LIBRARYNAME := clangStaticAnalyzerCheckers

BUILT_SOURCES = Checkers.inc
TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/Checkers.inc.tmp : Checkers.td $(PROJ_SRC_DIR)/$(CLANG_LEVEL)/include/clang/StaticAnalyzer/Checkers/CheckerBase.td $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang SA Checkers tables with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-sa-checkers -I $(PROJ_SRC_DIR)/$(CLANG_LEVEL)/include -o $(call SYSPATH, $@) $<
