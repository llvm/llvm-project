#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fa20  ########


fa20: run
	

build:  $(SRC)/fa20.f
	-$(RM) fa20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fa20.f -o fa20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fa20.$(OBJX) check.$(OBJX) $(LIBS) -o fa20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fa20
	fa20.$(EXESUFFIX)

verify: ;

fa20.run: run

