#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee15  ########


ieee15: ieee15.$(OBJX)
	

ieee15.$(OBJX):  $(SRC)/ieee15.f90
	-$(RM) ieee15.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) -Mpreprocess $(LDFLAGS) $(SRC)/ieee15.f90 -o ieee15.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee15.$(OBJX) check.$(OBJX) $(LIBS) -o ieee15.$(EXESUFFIX)


ieee15.run: ieee15.$(OBJX)
	@echo ------------------------------------ executing test ieee15
	ieee15.$(EXESUFFIX)
run: ieee15.$(OBJX)
	@echo ------------------------------------ executing test ieee15
	ieee15.$(EXESUFFIX)

build:	ieee15.$(OBJX)


verify:	;
