##===- clang/lib/Sema/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
#  This implements the semantic analyzer and AST builder library for the 
#  C-Language front-end.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../../..
LIBRARYNAME := clangSema
BUILD_ARCHIVE = 1
CXXFLAGS = -fno-rtti

# Don't install Clang libraries
NO_INSTALL = 1

CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include

include $(LEVEL)/Makefile.common

