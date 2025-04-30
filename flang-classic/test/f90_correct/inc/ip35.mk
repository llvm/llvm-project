#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip35  ########


ip35: run
	

build:  $(SRC)/ip35.f90
	-$(RM) ip35.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip35.f90 -o ip35.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip35.$(OBJX) check.$(OBJX) $(LIBS) -o ip35.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip35
	ip35.$(EXESUFFIX)

verify: ;

ip35.run: run

