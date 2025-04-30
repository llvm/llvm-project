#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test al06  ########


al06: run
	

build:  $(SRC)/al06.f90
	-$(RM) al06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/al06.f90 -o al06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) al06.$(OBJX) check.$(OBJX) $(LIBS) -o al06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test al06
	al06.$(EXESUFFIX)

verify: ;

al06.run: run

