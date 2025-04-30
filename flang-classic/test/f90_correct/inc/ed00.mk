#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ed00  ########


ed00: run
	

build:  $(SRC)/ed00.f
	-$(RM) ed00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ed00.f -o ed00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ed00.$(OBJX) check.$(OBJX) $(LIBS) -o ed00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ed00
	ed00.$(EXESUFFIX)

verify: ;

ed00.run: run

