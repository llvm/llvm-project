#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop413  ########


oop413: run
	

build:  $(SRC)/oop413.f90
	-$(RM) oop413.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop413.f90 -o oop413.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop413.$(OBJX) check.$(OBJX) $(LIBS) -o oop413.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop413
	oop413.$(EXESUFFIX)

verify: ;

