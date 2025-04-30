#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test md04  ########


md04: run
	

build:  $(SRC)/md04.f90
	-$(RM) md04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/md04.f90 -o md04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) md04.$(OBJX) check.$(OBJX) $(LIBS) -o md04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test md04
	md04.$(EXESUFFIX)

verify: ;

md04.run: run

