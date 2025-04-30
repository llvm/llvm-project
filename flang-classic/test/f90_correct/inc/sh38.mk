#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh38  ########


sh38: run
	

build:  $(SRC)/sh38.f90
	-$(RM) sh38.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh38.f90 -o sh38.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh38.$(OBJX) check.$(OBJX) $(LIBS) -o sh38.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh38
	sh38.$(EXESUFFIX)

verify: ;

sh38.run: run

