#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh21  ########


sh21: run
	

build:  $(SRC)/sh21.f90
	-$(RM) sh21.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh21.f90 -o sh21.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh21.$(OBJX) check.$(OBJX) $(LIBS) -o sh21.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh21
	sh21.$(EXESUFFIX)

verify: ;

sh21.run: run

