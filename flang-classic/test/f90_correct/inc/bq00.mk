#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bq00  ########


bq00: run
	

build:  $(SRC)/bq00.f
	-$(RM) bq00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bq00.f -o bq00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bq00.$(OBJX) check.$(OBJX) $(LIBS) -o bq00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bq00
	bq00.$(EXESUFFIX)

verify: ;

bq00.run: run

