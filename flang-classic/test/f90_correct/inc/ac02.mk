#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ac02  ########


ac02: run
	

build:  $(SRC)/ac02.f90
	-$(RM) ac02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ac02.f90 -o ac02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ac02.$(OBJX) check.$(OBJX) $(LIBS) -o ac02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ac02
	ac02.$(EXESUFFIX)

verify: ;

ac02.run: run

