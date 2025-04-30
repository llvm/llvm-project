#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee14  ########


ieee14: ieee14.$(OBJX)
	

ieee14.$(OBJX):  $(SRC)/ieee14.f90
	-$(RM) ieee14.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee14.f90 -o ieee14.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee14.$(OBJX) check.$(OBJX) $(LIBS) -o ieee14.$(EXESUFFIX)


ieee14.run: ieee14.$(OBJX)
	@echo ------------------------------------ executing test ieee14
	ieee14.$(EXESUFFIX)

verify:	;
build:	ieee14.$(OBJX) ;
run: ieee14.$(OBJX)
	@echo ------------------------------------ executing test ieee14
	ieee14.$(EXESUFFIX)
