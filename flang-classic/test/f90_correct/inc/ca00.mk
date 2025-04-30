#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ca00  ########


ca00: run
	

build:  $(SRC)/ca00.f
	-$(RM) ca00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ca00.f -o ca00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ca00.$(OBJX) check.$(OBJX) $(LIBS) -o ca00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ca00
	ca00.$(EXESUFFIX)

verify: ;

ca00.run: run

