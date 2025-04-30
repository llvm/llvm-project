#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test wa05  ########


wa05: run
	

build:  $(SRC)/wa05.f90
	-$(RM) wa05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/wa05.f90 -o wa05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) wa05.$(OBJX) check.$(OBJX) $(LIBS) -o wa05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test wa05
	wa05.$(EXESUFFIX)

verify: ;

wa05.run: run

