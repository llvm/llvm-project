#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ge02  ########


ge02: run
	

build:  $(SRC)/ge02.f90
	-$(RM) ge02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ge02.f90 -o ge02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ge02.$(OBJX) check.$(OBJX) $(LIBS) -o ge02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ge02
	ge02.$(EXESUFFIX)

verify: ;

ge02.run: run

