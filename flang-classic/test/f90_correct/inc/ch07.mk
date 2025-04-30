#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ch07  ########


ch07: run
	

build:  $(SRC)/ch07.f90
	-$(RM) ch07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ch07.f90 -o ch07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ch07.$(OBJX) check.$(OBJX) $(LIBS) -o ch07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ch07
	ch07.$(EXESUFFIX)

verify: ;

ch07.run: run

