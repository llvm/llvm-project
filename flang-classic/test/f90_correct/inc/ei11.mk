#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ei11  ########


ei11: run
	

build:  $(SRC)/ei11.f90
	-$(RM) ei11.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ei11.f90 -o ei11.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ei11.$(OBJX) check.$(OBJX) $(LIBS) -o ei11.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ei11
	ei11.$(EXESUFFIX)

verify: ;

ei11.run: run

