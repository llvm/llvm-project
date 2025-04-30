#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip02  ########


ip02: run
	

build:  $(SRC)/ip02.f90
	-$(RM) ip02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip02.f90 -o ip02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip02.$(OBJX) check.$(OBJX) $(LIBS) -o ip02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip02
	ip02.$(EXESUFFIX)

verify: ;

ip02.run: run

