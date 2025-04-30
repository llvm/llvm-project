#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh37  ########


sh37: run
	

build:  $(SRC)/sh37.f90
	-$(RM) sh37.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh37.f90 -o sh37.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh37.$(OBJX) check.$(OBJX) $(LIBS) -o sh37.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh37
	sh37.$(EXESUFFIX)

verify: ;

sh37.run: run

