#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test md02  ########


md02: run
	

build:  $(SRC)/md02.f90
	-$(RM) md02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/md02.f90 -o md02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) md02.$(OBJX) check.$(OBJX) $(LIBS) -o md02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test md02
	md02.$(EXESUFFIX)

verify: ;

md02.run: run

