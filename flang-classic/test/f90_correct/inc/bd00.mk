#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bd00  ########


bd00: run
	

build:  $(SRC)/bd00.f
	-$(RM) bd00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bd00.f -o bd00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bd00.$(OBJX) check.$(OBJX) $(LIBS) -o bd00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bd00
	bd00.$(EXESUFFIX)

verify: ;

bd00.run: run

