#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test al01  ########


al01: run
	

build:  $(SRC)/al01.f90
	-$(RM) al01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) -Mallocatable=03 $(LDFLAGS) $(SRC)/al01.f90 -o al01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) al01.$(OBJX) check.$(OBJX) $(LIBS) -o al01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test al01
	al01.$(EXESUFFIX)

verify: ;

al01.run: run

