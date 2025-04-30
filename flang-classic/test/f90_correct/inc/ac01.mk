#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ac01  ########


ac01: run
	

build:  $(SRC)/ac01.f
	-$(RM) ac01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ac01.f -o ac01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ac01.$(OBJX) check.$(OBJX) $(LIBS) -o ac01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ac01
	ac01.$(EXESUFFIX)

verify: ;

ac01.run: run

