#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ei10  ########


ei10: run
	

build:  $(SRC)/ei10.F90
	-$(RM) ei10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ei10.F90 -o ei10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ei10.$(OBJX) check.$(OBJX) $(LIBS) -o ei10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ei10
	ei10.$(EXESUFFIX)

verify: ;

ei10.run: run

