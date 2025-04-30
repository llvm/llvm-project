#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bq10  ########


bq10: run
	

build:  $(SRC)/bq10.f
	-$(RM) bq10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bq10.f -o bq10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bq10.$(OBJX) check.$(OBJX) $(LIBS) -o bq10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bq10
	bq10.$(EXESUFFIX)

verify: ;

bq10.run: run

