#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh22  ########


sh22: run
	

build:  $(SRC)/sh22.f90
	-$(RM) sh22.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh22.f90 -o sh22.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh22.$(OBJX) check.$(OBJX) $(LIBS) -o sh22.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh22
	sh22.$(EXESUFFIX)

verify: ;

sh22.run: run

