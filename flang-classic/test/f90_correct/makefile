#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception



BASE_DIR=$(HOMEQA)
SRC=$(BASE_DIR)/src
SRC2=$(BASE_DIR)/src/src  # For regression_cpu
FC=flang
CC=clang
CXX=clang++
OBJX=o
EXESUFFIX=exe
LD=$(FC)
OPT=-O
FFLAGS=-I$(SRC) $(OPT) $(KIEE) $(EXTRA_FFLAGS) $(EXTRA_HFLAGS)
LDFLAGS=$(EXTRA_LDFLAGS)
LIBS=$(EXTRA_LIBS)
CFLAGS=$(EXTRA_CFLAGS)
TEST=
COMP_CHECK=python $(HOMEQA)/../tools/check_compilation.py

RM=rm -f
CP=cp -f
CD=cd
MKDIR=mkdir -p
UNAME := $(shell uname -a)

INCLUDES=$(BASE_DIR)/inc

check: check.$(OBJX)

check.$(OBJX) : $(SRC)/check.c
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)

clean.run: clean.$(OBJX)
	a.out

clean:
	-$(RM) a.out *.$(OBJX) *.mod *.qdbg core

run.run: run.$(OBJX)
	a.out

include $(INCLUDES)/$(TEST).mk
