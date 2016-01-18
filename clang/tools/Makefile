##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ..

include $(CLANG_LEVEL)/../../Makefile.config

DIRS := 
PARALLEL_DIRS := clang-format driver diagtool

ifeq ($(ENABLE_CLANG_STATIC_ANALYZER), 1)
  PARALLEL_DIRS += clang-check
endif

ifeq ($(ENABLE_CLANG_ARCMT), 1)
  DIRS += libclang c-index-test c-arcmt-test
  PARALLEL_DIRS += arcmt-test
endif

# Recurse into the extra repository of tools if present.
OPTIONAL_PARALLEL_DIRS := extra

ifeq ($(BUILD_CLANG_ONLY),YES)
  DIRS := libclang c-index-test
  PARALLEL_DIRS := driver
  OPTIONAL_PARALLEL_DIRS :=
endif

include $(CLANG_LEVEL)/Makefile
