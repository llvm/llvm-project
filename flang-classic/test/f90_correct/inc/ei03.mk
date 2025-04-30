#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ei03  ########


ei03: run
	

build:  $(SRC)/ei03.f90
	-$(RM) ei03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ei03.f90 -o ei03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ei03.$(OBJX) check.$(OBJX) $(LIBS) -o ei03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ei03
	ei03.$(EXESUFFIX)

verify: ;

ei03.run: run

