#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip42  ########


ip42: run
	

build:  $(SRC)/ip42.f90
	-$(RM) ip42.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip42.f90 -o ip42.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip42.$(OBJX) check.$(OBJX) $(LIBS) -o ip42.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip42
	ip42.$(EXESUFFIX)

verify: ;

ip42.run: run

