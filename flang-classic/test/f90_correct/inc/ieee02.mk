#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee02  ########


ieee02: ieee02.$(OBJX)
	

ieee02.$(OBJX):  $(SRC)/ieee02.f90
	-$(RM) ieee02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee02.f90 -o ieee02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee02.$(OBJX) check.$(OBJX) $(LIBS) -o ieee02.$(EXESUFFIX)


ieee02.run: ieee02.$(OBJX)
	@echo ------------------------------------ executing test ieee02
	ieee02.$(EXESUFFIX)
run: ieee02.$(OBJX)
	@echo ------------------------------------ executing test ieee02
	ieee02.$(EXESUFFIX)

build:	ieee02.$(OBJX)
verify:	;
