#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception



BASE_DIR=$(HOMEQA)
SRC=$(BASE_DIR)/src
SRC2=$(BASE_DIR)/src/src  # For regression_cpu
FC=flang
F90=flang
CC=clang
CXX=clang++
OBJX=o
EXESUFFIX=exe
LD=$(FC)
OPT=-O
MPFLAGS=-fopenmp
CMNFLAGS=-c -I$(SRC) $(OPT) $(MPFLAGS)
FFLAGS=$(CMNFLAGS) $(KIEE) $(EXTRA_FFLAGS) $(EXTRA_HFLAGS)
LDFLAGS=$(EXTRA_LDFLAGS) $(OPT) $(MPFLAGS)
LIBS=$(EXTRA_LIBS)
CFLAGS=$(CMNFLAGS) $(EXTRA_CFLAGS)
TEST=

RUN=
RUN1=OMP_NUM_THREADS=1
RUN2=OMP_NUM_THREADS=2
RUN4=OMP_NUM_THREADS=4
RUN12=OMP_NUM_THREADS=12
RUN65=OMP_NUM_THREADS=65
NGREP=sh $(HOMEQA)/../tools/ngrep.sh

RM=rm -f
CP=cp -f

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

verify: ;
