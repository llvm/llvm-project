#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test wa02  ########


wa02: run
	

build:  $(SRC)/wa02.f90
	-$(RM) wa02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/wa02.f90 -o wa02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) wa02.$(OBJX) check.$(OBJX) $(LIBS) -o wa02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test wa02
	wa02.$(EXESUFFIX)

verify: ;

wa02.run: run

