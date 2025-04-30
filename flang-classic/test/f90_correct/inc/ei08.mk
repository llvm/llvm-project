#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ei08  ########


ei08: run
	

build:  $(SRC)/ei08.f90
	-$(RM) ei08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ei08.f90 -o ei08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ei08.$(OBJX) check.$(OBJX) $(LIBS) -o ei08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ei08
	ei08.$(EXESUFFIX)

verify: ;

ei08.run: run

