#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fa19  ########


fa19: run
	

build:  $(SRC)/fa19.f
	-$(RM) fa19.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fa19.f -o fa19.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fa19.$(OBJX) check.$(OBJX) $(LIBS) -o fa19.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fa19
	fa19.$(EXESUFFIX)

verify: ;

fa19.run: run

