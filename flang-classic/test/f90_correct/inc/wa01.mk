#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test wa01  ########


wa01: run
	

build:  $(SRC)/wa01.f90
	-$(RM) wa01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/wa01.f90 -o wa01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) wa01.$(OBJX) check.$(OBJX) $(LIBS) -o wa01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test wa01
	wa01.$(EXESUFFIX)

verify: ;

wa01.run: run

