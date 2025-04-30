#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt34  ########


dt34: run
	

build:  $(SRC)/dt34.f90
	-$(RM) dt34.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt34.f90 -o dt34.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt34.$(OBJX) check.$(OBJX) $(LIBS) -o dt34.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt34
	dt34.$(EXESUFFIX)

verify: ;

dt34.run: run

