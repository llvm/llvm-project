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

DIRS := driver libclang c-index-test arcmt-test c-arcmt-test diagtool \
        clang-check

# Recurse into the extra repository of tools if present.
OPTIONAL_DIRS := extra

ifeq ($(BUILD_CLANG_ONLY),YES)
  DIRS := driver libclang c-index-test
  OPTIONAL_DIRS :=
endif

include $(CLANG_LEVEL)/Makefile
