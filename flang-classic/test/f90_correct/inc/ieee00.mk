#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee00  ########


ieee00: ieee00.$(OBJX)
	

ieee00.$(OBJX):  $(SRC)/ieee00.f90
	-$(RM) ieee00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee00.f90 -o ieee00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee00.$(OBJX) check.$(OBJX) $(LIBS) -o ieee00.$(EXESUFFIX)


ieee00.run: ieee00.$(OBJX)
	@echo ------------------------------------ executing test ieee00
	ieee00.$(EXESUFFIX)
run: ieee00.$(OBJX)
	@echo ------------------------------------ executing test ieee00
	ieee00.$(EXESUFFIX)

verify:	;
build:	ieee00.$(OBJX)
