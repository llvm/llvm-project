#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nu01  ########


nu01: run
	

build:  $(SRC)/nu01.f90
	-$(RM) nu01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nu01.f90 -o nu01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nu01.$(OBJX) check.$(OBJX) $(LIBS) -o nu01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nu01
	nu01.$(EXESUFFIX)

verify: ;

nu01.run: run

