#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bq34  ########


bq34: run
	

build:  $(SRC)/bq34.f
	-$(RM) bq34.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bq34.f -o bq34.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bq34.$(OBJX) check.$(OBJX) $(LIBS) -o bq34.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bq34
	bq34.$(EXESUFFIX)

verify: ;

bq34.run: run

