#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test al24  ########


al24: run
	

build:  $(SRC)/al24.f90
	-$(RM) al24.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/al24.f90 -o al24.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) al24.$(OBJX) check.$(OBJX) $(LIBS) -o al24.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test al24
	al24.$(EXESUFFIX)

verify: ;

