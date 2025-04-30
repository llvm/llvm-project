#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bs01  ########


bs01: run
	

build:  $(SRC)/bs01.f
	-$(RM) bs01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bs01.f -o bs01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bs01.$(OBJX) check.$(OBJX) $(LIBS) -o bs01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bs01
	bs01.$(EXESUFFIX)

verify: ;

bs01.run: run

