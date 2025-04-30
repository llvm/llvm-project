#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh34  ########


sh34: run
	

build:  $(SRC)/sh34.f90
	-$(RM) sh34.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh34.f90 -o sh34.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh34.$(OBJX) check.$(OBJX) $(LIBS) -o sh34.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh34
	sh34.$(EXESUFFIX)

verify: ;

sh34.run: run

