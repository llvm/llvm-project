#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fc20  ########


fc20: run
	

build:  $(SRC)/fc20.f
	-$(RM) fc20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fc20.f -o fc20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fc20.$(OBJX) check.$(OBJX) $(LIBS) -o fc20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fc20
	fc20.$(EXESUFFIX)

verify: ;

fc20.run: run

