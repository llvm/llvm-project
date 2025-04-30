#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test rn00  ########


rn00: run
	

build:  $(SRC)/rn00.f90
	-$(RM) rn00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/rn00.f90 -o rn00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) rn00.$(OBJX) check.$(OBJX) $(LIBS) -o rn00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test rn00
	rn00.$(EXESUFFIX)

verify: ;

rn00.run: run

