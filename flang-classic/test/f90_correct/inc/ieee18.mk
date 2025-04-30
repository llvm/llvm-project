#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee18  ########

ieee18: ieee18.$(OBJX)
	

ieee18.$(OBJX):  $(SRC)/ieee18.f90
	-$(RM) ieee18.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee18.f90 -o ieee18.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee18.$(OBJX) check.$(OBJX) $(LIBS) -o ieee18.$(EXESUFFIX)


ieee18.run: ieee18.$(OBJX)
	@echo ------------------------------------ executing test ieee18
	ieee18.$(EXESUFFIX)

verify: ;
build: ieee18.$(OBJX) ;
run: ieee18.run ;
