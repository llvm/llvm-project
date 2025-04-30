#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test al08  ########


al08: run
	

build:  $(SRC)/al08.f90
	-$(RM) al08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) -Mchkptr $(LDFLAGS) $(SRC)/al08.f90 -o al08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) al08.$(OBJX) check.$(OBJX) $(LIBS) -o al08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test al08
	al08.$(EXESUFFIX)

verify: ;

al08.run: run

