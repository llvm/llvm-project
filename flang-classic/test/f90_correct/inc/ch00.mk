#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ch00  ########


ch00: run
	

build:  $(SRC)/ch00.f90
	-$(RM) ch00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ch00.f90 -o ch00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ch00.$(OBJX) check.$(OBJX) $(LIBS) -o ch00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ch00
	ch00.$(EXESUFFIX)

verify: ;

ch00.run: run

