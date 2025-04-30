#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
SRC2 = $(SRC)/src
FFLAGS += -Mpreprocess

build: check.$(OBJX)
	-$(FC) $(FFLAGS) $(SRC2)/ci03.f90 -I$(SRC2) -o ci03.$(OBJX)
	-$(FC) $(MPFLAGS) ci03.$(OBJX) $(OPT) check.$(OBJX) -I$(SRC2) -o ci03.$(EXE)

run:
	ci03.$(EXE)
