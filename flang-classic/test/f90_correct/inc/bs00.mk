#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bs00  ########


bs00: run
	

build:  $(SRC)/bs00.f
	-$(RM) bs00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bs00.f -o bs00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bs00.$(OBJX) check.$(OBJX) $(LIBS) -o bs00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bs00
	bs00.$(EXESUFFIX)

verify: ;

bs00.run: run

