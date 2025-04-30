#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nu02  ########


nu02: run
	

build:  $(SRC)/nu02.f90
	-$(RM) nu02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nu02.f90 -o nu02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nu02.$(OBJX) check.$(OBJX) $(LIBS) -o nu02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nu02
	nu02.$(EXESUFFIX)

verify: ;

nu02.run: run

