#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fa10  ########


fa10: run
	

build:  $(SRC)/fa10.f
	-$(RM) fa10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fa10.f -o fa10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fa10.$(OBJX) check.$(OBJX) $(LIBS) -o fa10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fa10
	fa10.$(EXESUFFIX)

verify: ;

fa10.run: run

