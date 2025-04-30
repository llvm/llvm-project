#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt01  ########


dt01: run
	

build:  $(SRC)/dt01.f90
	-$(RM) dt01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt01.f90 -o dt01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt01.$(OBJX) check.$(OBJX) $(LIBS) -o dt01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt01
	dt01.$(EXESUFFIX)

verify: ;

dt01.run: run

