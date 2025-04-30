#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ch23  ########


ch23: run
	

build:  $(SRC)/ch23.f90
	-$(RM) ch23.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ch23.f90 -o ch23.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ch23.$(OBJX) check.$(OBJX) $(LIBS) -o ch23.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ch23
	ch23.$(EXESUFFIX)

verify: ;

ch23.run: run

