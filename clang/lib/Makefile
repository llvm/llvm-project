##===- lib/Makefile ----------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
CLANG_LEVEL := ..

# ARCMigrate and Rewrite are always needed because of libclang.
PARALLEL_DIRS = Headers Basic APINotes Lex Parse AST Sema CodeGen Analysis \
                Frontend FrontendTool Tooling Driver Format Edit Rewrite \
                Serialization Index ASTMatchers

include $(CLANG_LEVEL)/../../Makefile.config

ifeq ($(ENABLE_CLANG_STATIC_ANALYZER),1)
PARALLEL_DIRS += StaticAnalyzer
endif

ifeq ($(ENABLE_CLANG_ARCMT),1)
PARALLEL_DIRS += ARCMigrate
endif

include $(CLANG_LEVEL)/Makefile
