#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test md17  ########


md17: run
	

build:  $(SRC)/md17.f90
	-$(RM) md17.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/md17.f90 -o md17.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) md17.$(OBJX) check.$(OBJX) $(LIBS) -o md17.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test md17
	md17.$(EXESUFFIX)

verify: ;

md17.run: run

