#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt18  ########


dt18: run
	

build:  $(SRC)/dt18.f90
	-$(RM) dt18.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt18.f90 -o dt18.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt18.$(OBJX) check.$(OBJX) $(LIBS) -o dt18.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt18
	dt18.$(EXESUFFIX)

verify: ;

dt18.run: run

