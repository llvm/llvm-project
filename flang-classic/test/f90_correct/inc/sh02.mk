#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh02  ########


sh02: run
	

build:  $(SRC)/sh02.f90
	-$(RM) sh02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh02.f90 -o sh02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh02.$(OBJX) check.$(OBJX) $(LIBS) -o sh02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh02
	sh02.$(EXESUFFIX)

verify: ;

sh02.run: run

