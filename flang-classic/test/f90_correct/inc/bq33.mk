#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bq33  ########


bq33: run
	

build:  $(SRC)/bq33.f
	-$(RM) bq33.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bq33.f -o bq33.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bq33.$(OBJX) check.$(OBJX) $(LIBS) -o bq33.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bq33
	bq33.$(EXESUFFIX)

verify: ;

bq33.run: run

