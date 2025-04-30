#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh01  ########


sh01: run
	

build:  $(SRC)/sh01.f90
	-$(RM) sh01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh01.f90 -o sh01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh01.$(OBJX) check.$(OBJX) $(LIBS) -o sh01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh01
	sh01.$(EXESUFFIX)

verify: ;

sh01.run: run

