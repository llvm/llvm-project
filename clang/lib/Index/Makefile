##===- clang/lib/Index/Makefile ----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
#  This implements the Indexer library for the C-Language front-end.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../../..
include $(LEVEL)/Makefile.config

LIBRARYNAME := clangIndex
BUILD_ARCHIVE = 1
CXXFLAGS = -fno-rtti 

# Don't install Clang libraries
NO_INSTALL = 1

ifeq ($(ARCH),PowerPC)
CXXFLAGS += -maltivec
endif

CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include

include $(LEVEL)/Makefile.common

