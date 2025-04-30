#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh31  ########


sh31: run
	

build:  $(SRC)/sh31.f90
	-$(RM) sh31.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh31.f90 -o sh31.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh31.$(OBJX) check.$(OBJX) $(LIBS) -o sh31.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh31
	sh31.$(EXESUFFIX)

verify: ;

sh31.run: run

