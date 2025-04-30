#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt06  ########


dt06: run
	

build:  $(SRC)/dt06.f90
	-$(RM) dt06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt06.f90 -o dt06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt06.$(OBJX) check.$(OBJX) $(LIBS) -o dt06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt06
	dt06.$(EXESUFFIX)

verify: ;

dt06.run: run

