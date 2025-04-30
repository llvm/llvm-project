#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee03  ########


ieee03: ieee03.$(OBJX)

ieee03.$(OBJX):  $(SRC)/ieee03.f90
	-$(RM) ieee03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee03.f90 -o ieee03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee03.$(OBJX) check.$(OBJX) $(LIBS) -o ieee03.$(EXESUFFIX)


ieee03.run: ieee03.$(OBJX)
	@echo ------------------------------------ executing test ieee03
	ieee03.$(EXESUFFIX)

verify: ;
build: ieee03.$(OBJX) ;
run: ieee03.run ;
