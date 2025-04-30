#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee01  ########


ieee01: ieee01.$(OBJX)
	

ieee01.$(OBJX):  $(SRC)/ieee01.f90
	-$(RM) ieee01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee01.f90 -o ieee01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee01.$(OBJX) check.$(OBJX) $(LIBS) -o ieee01.$(EXESUFFIX)


ieee01.run: ieee01.$(OBJX)
	@echo ------------------------------------ executing test ieee01
	ieee01.$(EXESUFFIX)
run: ieee01.$(OBJX)
	@echo ------------------------------------ executing test ieee01
	ieee01.$(EXESUFFIX)

verify:	;
build:	ieee01.$(OBJX)
