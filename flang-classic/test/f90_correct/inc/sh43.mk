#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh43  ########


sh43: run
	

build:  $(SRC)/sh43.f90
	-$(RM) sh43.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh43.f90 -o sh43.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh43.$(OBJX) check.$(OBJX) $(LIBS) -o sh43.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh43
	sh43.$(EXESUFFIX)

verify: ;

sh43.run: run

