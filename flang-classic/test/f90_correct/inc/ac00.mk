#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ac00  ########


ac00: run
	

build:  $(SRC)/ac00.f
	-$(RM) ac00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ac00.f -o ac00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ac00.$(OBJX) check.$(OBJX) $(LIBS) -o ac00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ac00
	ac00.$(EXESUFFIX)

verify: ;

ac00.run: run

