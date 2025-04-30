#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test al00  ########


al00: run
	

build:  $(SRC)/al00.f90
	-$(RM) al00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) -Mallocatable=03 $(LDFLAGS) $(SRC)/al00.f90 -o al00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) al00.$(OBJX) check.$(OBJX) $(LIBS) -o al00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test al00
	al00.$(EXESUFFIX)

verify: ;

al00.run: run

