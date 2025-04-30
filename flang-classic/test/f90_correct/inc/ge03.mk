#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ge03  ########


ge03: run
	

build:  $(SRC)/ge03.f90
	-$(RM) ge03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ge03.f90 -o ge03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ge03.$(OBJX) check.$(OBJX) $(LIBS) -o ge03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ge03
	ge03.$(EXESUFFIX)

verify: ;

ge03.run: run

