#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh35  ########


sh35: run
	

build:  $(SRC)/sh35.f90
	-$(RM) sh35.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh35.f90 -o sh35.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh35.$(OBJX) check.$(OBJX) $(LIBS) -o sh35.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh35
	sh35.$(EXESUFFIX)

verify: ;

sh35.run: run

