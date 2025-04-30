#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bq35  ########


bq35: run
	

build:  $(SRC)/bq35.f
	-$(RM) bq35.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bq35.f -o bq35.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bq35.$(OBJX) check.$(OBJX) $(LIBS) -o bq35.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bq35
	bq35.$(EXESUFFIX)

verify: ;

bq35.run: run

