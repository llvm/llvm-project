#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ei28  ########


ei28: run
	

build:  $(SRC)/ei28.f90
	-$(RM) ei28.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ei28.f90 -o ei28.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ei28.$(OBJX) check.$(OBJX) $(LIBS) -o ei28.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ei28
	ei28.$(EXESUFFIX)

verify: ;

ei28.run: run

