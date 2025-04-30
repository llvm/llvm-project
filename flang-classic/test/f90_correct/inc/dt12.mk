#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt12  ########


dt12: run
	

build:  $(SRC)/dt12.f90
	-$(RM) dt12.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt12.f90 -o dt12.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt12.$(OBJX) check.$(OBJX) $(LIBS) -o dt12.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt12
	dt12.$(EXESUFFIX)

verify: ;

dt12.run: run

