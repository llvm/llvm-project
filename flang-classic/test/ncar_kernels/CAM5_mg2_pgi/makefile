#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# makefile for NCAR Kernel tests.


#TEST_DIR=./src
SRC=$(HOMEQA)
INCLUDES = $(HOMEQA)/inc
SRC_DIR=$(SRC)/src
FC=flang
OBJX=o
EXTRA_CFLAGS=
EXTRA_FFLAGS=
LD=$(FC)
RUN=
OPT=
ENDIAN=
FFLAGS=$(OPT)
LDFLAGS=$(EXTRA_LDFLAGS)
LIBS=$(EXTRA_LIBS)
KIEE=
CFLAGS=$(OPT) $(EXTRA_CFLAGS) 
EXE=out


RM=rm -f

TEST = t1
include $(INCLUDES)/$(TEST).mk

