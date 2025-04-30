#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test wa06  ########


wa06: run
	

build:  $(SRC)/wa06.f90
	-$(RM) wa06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/wa06.f90 -o wa06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) wa06.$(OBJX) check.$(OBJX) $(LIBS) -o wa06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test wa06
	wa06.$(EXESUFFIX)

verify: ;

wa06.run: run

