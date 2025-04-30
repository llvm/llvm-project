#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test md16  ########


md16: run
	

build:  $(SRC)/md16.f90
	-$(RM) md16.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/md16.f90 -o md16.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) md16.$(OBJX) check.$(OBJX) $(LIBS) -o md16.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test md16
	md16.$(EXESUFFIX)

verify: ;

md16.run: run

