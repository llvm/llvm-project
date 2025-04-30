#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh11  ########


sh11: run
	

build:  $(SRC)/sh11.f90
	-$(RM) sh11.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh11.f90 -o sh11.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh11.$(OBJX) check.$(OBJX) $(LIBS) -o sh11.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh11
	sh11.$(EXESUFFIX)

verify: ;

sh11.run: run

