#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip17  ########


ip17: run
	

build:  $(SRC)/ip17.f90
	-$(RM) ip17.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip17.f90 -o ip17.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip17.$(OBJX) check.$(OBJX) $(LIBS) -o ip17.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip17
	ip17.$(EXESUFFIX)

verify: ;

ip17.run: run

