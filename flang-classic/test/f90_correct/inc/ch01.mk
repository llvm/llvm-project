#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ch01  ########


ch01: run
	

build:  $(SRC)/ch01.f90
	-$(RM) ch01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ch01.f90 -o ch01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ch01.$(OBJX) check.$(OBJX) $(LIBS) -o ch01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ch01
	ch01.$(EXESUFFIX)

verify: ;

ch01.run: run

