#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt07  ########


dt07: run
	

build:  $(SRC)/dt07.f90
	-$(RM) dt07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt07.f90 -o dt07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt07.$(OBJX) check.$(OBJX) $(LIBS) -o dt07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt07
	dt07.$(EXESUFFIX)

verify: ;

dt07.run: run

