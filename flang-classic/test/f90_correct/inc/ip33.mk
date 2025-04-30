#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip33  ########


ip33: run
	

build:  $(SRC)/ip33.f90
	-$(RM) ip33.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip33.f90 -o ip33.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip33.$(OBJX) check.$(OBJX) $(LIBS) -o ip33.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip33
	ip33.$(EXESUFFIX)

verify: ;

ip33.run: run

