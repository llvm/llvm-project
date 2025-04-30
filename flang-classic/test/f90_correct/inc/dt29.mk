#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt29  ########


dt29: run
	

build:  $(SRC)/dt29.f90
	-$(RM) dt29.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt29.f90 -o dt29.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt29.$(OBJX) check.$(OBJX) $(LIBS) -o dt29.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt29
	dt29.$(EXESUFFIX)

verify: ;

dt29.run: run

