#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee13  ########


ieee13: ieee13.$(OBJX)
	

ieee13.$(OBJX):  $(SRC)/ieee13.f90
	-$(RM) ieee13.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee13.f90 -o ieee13.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee13.$(OBJX) check.$(OBJX) $(LIBS) -o ieee13.$(EXESUFFIX)


ieee13.run: ieee13.$(OBJX)
	@echo ------------------------------------ executing test ieee13
	ieee13.$(EXESUFFIX)

verify: ;
build: ieee13.$(OBJX) ;
run: ieee13.run ;
