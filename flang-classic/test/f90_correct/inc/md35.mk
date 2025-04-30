#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test md35  ########


md35: run
	

build:  $(SRC)/md35.f90
	-$(RM) md35.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/md35.f90 -o md35.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) md35.$(OBJX) check.$(OBJX) $(LIBS) -o md35.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test md35
	md35.$(EXESUFFIX)

verify: ;

md35.run: run

