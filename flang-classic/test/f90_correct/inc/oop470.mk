#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop470  ########


oop470: run
	

build:  $(SRC)/oop470.f90
	-$(RM) oop470.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop470.f90 -o oop470.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop470.$(OBJX) check.$(OBJX) $(LIBS) -o oop470.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop470
	oop470.$(EXESUFFIX)

verify: ;

