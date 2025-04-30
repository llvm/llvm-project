#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh36  ########


sh36: run
	

build:  $(SRC)/sh36.f90
	-$(RM) sh36.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh36.f90 -o sh36.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh36.$(OBJX) check.$(OBJX) $(LIBS) -o sh36.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh36
	sh36.$(EXESUFFIX)

verify: ;

sh36.run: run

