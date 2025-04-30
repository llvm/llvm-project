#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee16  ########


ieee16: ieee16.$(OBJX)
	

ieee16.$(OBJX):  $(SRC)/ieee16.f90
	-$(RM) ieee16.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee16.f90 -o ieee16.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee16.$(OBJX) check.$(OBJX) $(LIBS) -o ieee16.$(EXESUFFIX)


ieee16.run: ieee16.$(OBJX)
	@echo ------------------------------------ executing test ieee16
	ieee16.$(EXESUFFIX)

verify: ;
build: ieee16.$(OBJX) ;
run: ieee16.run ;
