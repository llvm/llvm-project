#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test md22  ########


md22: run
	

build:  $(SRC)/md22.f90
	-$(RM) md22.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/md22.f90 -o md22.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) md22.$(OBJX) check.$(OBJX) $(LIBS) -o md22.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test md22
	md22.$(EXESUFFIX)

verify: ;

md22.run: run

