#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test al02  ########


al02: run
	

build:  $(SRC)/al02.f90
	-$(RM) al02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) -Mallocatable=03 $(LDFLAGS) $(SRC)/al02.f90 -o al02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) al02.$(OBJX) check.$(OBJX) $(LIBS) -o al02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test al02
	al02.$(EXESUFFIX)

verify: ;

al02.run: run

