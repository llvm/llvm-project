#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ch22  ########


ch22: run
	

build:  $(SRC)/ch22.f
	-$(RM) ch22.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ch22.f -o ch22.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ch22.$(OBJX) check.$(OBJX) $(LIBS) -o ch22.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ch22
	ch22.$(EXESUFFIX)

verify: ;

ch22.run: run

