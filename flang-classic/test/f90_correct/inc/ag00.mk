#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ag00  ########


ag00: run
	

build:  $(SRC)/ag00.f
	-$(RM) ag00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ag00.f -o ag00.$(OBJX) -Mstandard
	-$(FC) $(FFLAGS) $(LDFLAGS) ag00.$(OBJX) check.$(OBJX) $(LIBS) -o ag00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ag00
	ag00.$(EXESUFFIX)

verify: ;

ag00.run: run

