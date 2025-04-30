#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt00  ########


dt00: run
	

build:  $(SRC)/dt00.f90
	-$(RM) dt00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt00.f90 -o dt00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt00.$(OBJX) check.$(OBJX) $(LIBS) -o dt00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt00
	dt00.$(EXESUFFIX)

verify: ;

dt00.run: run

