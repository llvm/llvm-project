#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fa00  ########


fa00: run
	

build:  $(SRC)/fa00.f
	-$(RM) fa00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fa00.f -Mpreprocess -o fa00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fa00.$(OBJX) check.$(OBJX) $(LIBS) -o fa00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fa00
	fa00.$(EXESUFFIX)

verify: ;

fa00.run: run

