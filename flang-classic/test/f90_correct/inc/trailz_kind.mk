#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE=trailz_kind.$(EXESUFFIX)

build:  $(SRC)/trailz_kind.f90
	-$(RM) trailz_kind.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/trailz_kind.f90 check.$(OBJX) -o trailz_kind.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test trailz_kind
	trailz_kind.$(EXESUFFIX)

verify: ;

trailz.run: run
