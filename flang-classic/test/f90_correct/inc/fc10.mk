#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fc10  ########


fc10: run
	

build:  $(SRC)/fc10.f
	-$(RM) fc10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fc10.f -o fc10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fc10.$(OBJX) check.$(OBJX) $(LIBS) -o fc10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fc10
	fc10.$(EXESUFFIX)

verify: ;

fc10.run: run

