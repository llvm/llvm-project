#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bq02  ########


bq02: run
	

build:  $(SRC)/bq02.f
	-$(RM) bq02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bq02.f -o bq02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bq02.$(OBJX) check.$(OBJX) $(LIBS) -o bq02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bq02
	bq02.$(EXESUFFIX)

verify: ;

bq02.run: run

