#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test md06  ########


md06: run
	

build:  $(SRC)/md06.f90
	-$(RM) md06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/md06.f90 -o md06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) md06.$(OBJX) check.$(OBJX) $(LIBS) -o md06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test md06
	md06.$(EXESUFFIX)

verify: ;

md06.run: run

