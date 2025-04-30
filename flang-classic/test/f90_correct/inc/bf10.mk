#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bf10  ########


bf10: run
	

build:  $(SRC)/bf10.f
	-$(RM) bf10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bf10.f -o bf10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bf10.$(OBJX) check.$(OBJX) $(LIBS) -o bf10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bf10
	bf10.$(EXESUFFIX)

verify: ;

bf10.run: run

