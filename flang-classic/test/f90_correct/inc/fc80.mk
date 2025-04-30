#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fc80  ########


fc80: run
	

build:  $(SRC)/fc80.f
	-$(RM) fc80.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fc80.f -o fc80.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fc80.$(OBJX) check.$(OBJX) $(LIBS) -o fc80.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fc80
	fc80.$(EXESUFFIX)

verify: ;

fc80.run: run

