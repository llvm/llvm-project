#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh24  ########


sh24: run
	

build:  $(SRC)/sh24.f90
	-$(RM) sh24.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh24.f90 -o sh24.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh24.$(OBJX) check.$(OBJX) $(LIBS) -o sh24.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh24
	sh24.$(EXESUFFIX)

verify: ;

sh24.run: run

