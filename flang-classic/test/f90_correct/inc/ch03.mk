#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ch03  ########


ch03: run
	

build:  $(SRC)/ch03.f90
	-$(RM) ch03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ch03.f90 -o ch03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ch03.$(OBJX) check.$(OBJX) $(LIBS) -o ch03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ch03
	ch03.$(EXESUFFIX)

verify: ;

ch03.run: run

