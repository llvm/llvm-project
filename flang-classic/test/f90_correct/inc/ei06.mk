#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ei06  ########


ei06: run
	

build:  $(SRC)/ei06.f90
	-$(RM) ei06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ei06.f90 -o ei06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ei06.$(OBJX) check.$(OBJX) $(LIBS) -o ei06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ei06
	ei06.$(EXESUFFIX)

verify: ;

ei06.run: run

