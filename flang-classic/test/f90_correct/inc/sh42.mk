#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh42  ########


sh42: run
	

build:  $(SRC)/sh42.f90
	-$(RM) sh42.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh42.f90 -o sh42.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh42.$(OBJX) check.$(OBJX) $(LIBS) -o sh42.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh42
	sh42.$(EXESUFFIX)

verify: ;

sh42.run: run

