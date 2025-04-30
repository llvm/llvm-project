#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt04  ########


dt04: run
	

build:  $(SRC)/dt04.f90
	-$(RM) dt04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt04.f90 -o dt04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt04.$(OBJX) check.$(OBJX) $(LIBS) -o dt04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt04
	dt04.$(EXESUFFIX)

verify: ;

dt04.run: run

