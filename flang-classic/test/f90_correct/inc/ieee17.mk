#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee17  ########


ieee17: ieee17.$(OBJX)
	

ieee17.$(OBJX):  $(SRC)/ieee17.f90
	-$(RM) ieee17.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee17.f90 -o ieee17.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee17.$(OBJX) check.$(OBJX) $(LIBS) -o ieee17.$(EXESUFFIX)


ieee17.run: ieee17.$(OBJX)
	@echo ------------------------------------ executing test ieee17
	ieee17.$(EXESUFFIX)
run: ieee17.$(OBJX)
	@echo ------------------------------------ executing test ieee17
	ieee17.$(EXESUFFIX)

build:	ieee17.$(OBJX)


verify:	;
