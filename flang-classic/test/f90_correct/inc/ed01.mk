#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ed01  ########


ed01: run
	

build:  $(SRC)/ed01.f
	-$(RM) ed01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ed01.f -o ed01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ed01.$(OBJX) check.$(OBJX) $(LIBS) -o ed01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ed01
	ed01.$(EXESUFFIX)

verify: ;

ed01.run: run

