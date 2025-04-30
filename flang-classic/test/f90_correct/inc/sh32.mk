#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh32  ########


sh32: run
	

build:  $(SRC)/sh32.f90
	-$(RM) sh32.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh32.f90 -o sh32.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh32.$(OBJX) check.$(OBJX) $(LIBS) -o sh32.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh32
	sh32.$(EXESUFFIX)

verify: ;

sh32.run: run

