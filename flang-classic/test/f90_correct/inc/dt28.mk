#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt28  ########


dt28: run
	

build:  $(SRC)/dt28.f90
	-$(RM) dt28.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt28.f90 -o dt28.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt28.$(OBJX) check.$(OBJX) $(LIBS) -o dt28.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt28
	dt28.$(EXESUFFIX)

verify: ;

dt28.run: run

