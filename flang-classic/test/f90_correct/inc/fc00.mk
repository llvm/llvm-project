#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fc00  ########


fc00: run
	

build:  $(SRC)/fc00.f
	-$(RM) fc00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fc00.f -o fc00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fc00.$(OBJX) check.$(OBJX) $(LIBS) -o fc00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fc00
	fc00.$(EXESUFFIX)

verify: ;

fc00.run: run

