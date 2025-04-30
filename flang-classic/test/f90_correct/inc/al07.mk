#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test al07  ########


al07: run
	

build:  $(SRC)/al07.f90
	-$(RM) al07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/al07.f90 -o al07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) al07.$(OBJX) check.$(OBJX) $(LIBS) -o al07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test al07
	al07.$(EXESUFFIX)

verify: ;

al07.run: run

