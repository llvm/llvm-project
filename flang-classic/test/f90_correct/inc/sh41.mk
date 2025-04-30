#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh41  ########


sh41: run
	

build:  $(SRC)/sh41.f90
	-$(RM) sh41.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh41.f90 -o sh41.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh41.$(OBJX) check.$(OBJX) $(LIBS) -o sh41.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh41
	sh41.$(EXESUFFIX)

verify: ;

sh41.run: run

