#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

 
BASE_DIR=$(HOMEQA)
SRC=$(BASE_DIR)
SRC_DIR=$(BASE_DIR)/openmp_examples
FC=flang
CC=clang
CXX=clang++
OBJX=o
EXESUFFIX=exe
EXE=out


EXTRA_CFLAGS=
EXTRA_FFLAGS=
LD=$(FC)
RUN=
OPT=-O
MPFLAGS=-mp
CMNFLAGS=-c -I$(SRC) $(OPT) $(MPFLAGS)
FFLAGS=$(CMNFLAGS) $(KIEE) $(EXTRA_FFLAGS) $(EXTRA_HFLAGS)
LDFLAGS=$(EXTRA_LDFLAGS) $(OPT) $(MPFLAGS)
LIBS=$(EXTRA_LIBS)
CFLAGS=$(CMNFLAGS) $(EXTRA_CFLAGS)
TEST = 
RM=rm -f
CP=cp -f
RUN=
RUN1=OMP_NUM_THREADS=1
RUN2=OMP_NUM_THREADS=2
RUN4=OMP_NUM_THREADS=4
RUN12=OMP_NUM_THREADS=12
RUN65=OMP_NUM_THREADS=65
COMP_CHECK = python $(BASE_DIR)/../tools/check_compilation.py

INCLUDES = $(SRC)/sources/inc
include $(INCLUDES)/$(TEST).mk

