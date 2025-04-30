#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt05  ########


dt05: run
	

build:  $(SRC)/dt05.f90
	-$(RM) dt05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt05.f90 -o dt05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt05.$(OBJX) check.$(OBJX) $(LIBS) -o dt05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt05
	dt05.$(EXESUFFIX)

verify: ;

dt05.run: run

