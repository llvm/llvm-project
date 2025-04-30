#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh12  ########


sh12: run
	

build:  $(SRC)/sh12.f90
	-$(RM) sh12.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh12.f90 -o sh12.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh12.$(OBJX) check.$(OBJX) $(LIBS) -o sh12.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh12
	sh12.$(EXESUFFIX)

verify: ;

sh12.run: run

