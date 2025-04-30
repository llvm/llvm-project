#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE=trailz_elemental.$(EXESUFFIX)

build:  $(SRC)/trailz_elemental.f90
	-$(RM) trailz_elemental.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/trailz_elemental.f90 check.$(OBJX) -o trailz_elemental.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test trailz_elemental
	trailz_elemental.$(EXESUFFIX)

verify: ;

trailz.run: run
