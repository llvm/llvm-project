#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt11  ########


dt11: run
	

build:  $(SRC)/dt11.f90
	-$(RM) dt11.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt11.f90 -o dt11.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt11.$(OBJX) check.$(OBJX) $(LIBS) -o dt11.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt11
	dt11.$(EXESUFFIX)

verify: ;

dt11.run: run

