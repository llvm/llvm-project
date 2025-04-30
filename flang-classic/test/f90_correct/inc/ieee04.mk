#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee04  ########


ieee04: ieee04.$(OBJX)

ieee04.$(OBJX):  $(SRC)/ieee04.f90
	-$(RM) ieee04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee04.f90 -o ieee04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee04.$(OBJX) check.$(OBJX) $(LIBS) -o ieee04.$(EXESUFFIX)


ieee04.run: ieee04.$(OBJX)
	@echo ------------------------------------ executing test ieee04
	ieee04.$(EXESUFFIX)

verify: ;
build: ieee04.$(OBJX) ;
run: ieee04.run ;
