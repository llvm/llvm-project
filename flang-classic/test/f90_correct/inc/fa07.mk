#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fa07  ########


fa07: run
	

build:  $(SRC)/fa07.f
	-$(RM) fa07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fa07.f -o fa07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fa07.$(OBJX) check.$(OBJX) $(LIBS) -o fa07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fa07
	fa07.$(EXESUFFIX)

verify: ;

fa07.run: run

