#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip36  ########


ip36: run
	

build:  $(SRC)/ip36.f90
	-$(RM) ip36.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip36.f90 -o ip36.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip36.$(OBJX) check.$(OBJX) $(LIBS) -o ip36.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip36
	ip36.$(EXESUFFIX)

verify: ;

ip36.run: run

