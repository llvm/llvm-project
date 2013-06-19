##===- source/Core/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../..
LIBRARYNAME := lldbCore
BUILD_ARCHIVE = 1

include $(LLDB_LEVEL)/Makefile

# Enable RTTI on GCC builds because one source file in this directory
# (cxa_demangle.cpp) uses dynamic_cast<> and GCC (at least 4.6 and 4.7)
# complain if we try to compile it with -fno-rtti.
$(info shell basename CXX is $(shell basename $(CXX)))
ifeq (g++,$(shell basename $(CXX) | colrm 4))
$(ObjDir)/cxa_demangle.o: Compile.CXX := $(filter-out -fno-rtti,$(Compile.CXX)) -frtti
endif

ifeq (Darwin,$(shell uname -s))
$(ObjDir)/cxa_demangle.o: Compile.CXX := $(filter-out -fno-rtti,$(Compile.CXX)) -frtti
endif
