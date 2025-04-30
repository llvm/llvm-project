#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fa06  ########


fa06: run
	

build:  $(SRC)/fa06.f
	-$(RM) fa06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fa06.f -o fa06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fa06.$(OBJX) check.$(OBJX) $(LIBS) -o fa06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fa06
	fa06.$(EXESUFFIX)

verify: ;

fa06.run: run

