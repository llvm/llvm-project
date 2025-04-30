#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ag01  ########


ag01: run
	

build:  $(SRC)/ag01.f90
	-$(RM) ag01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ag01.f90 -o ag01.$(OBJX) -Mstandard
	-$(FC) $(FFLAGS) $(LDFLAGS) ag01.$(OBJX) check.$(OBJX) $(LIBS) -o ag01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ag01
	ag01.$(EXESUFFIX)

verify: ;

ag01.run: run

