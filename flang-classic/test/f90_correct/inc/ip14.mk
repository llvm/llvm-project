#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip14  ########


ip14: run
	

build:  $(SRC)/ip14.f90
	-$(RM) ip14.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip14.f90 -o ip14.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip14.$(OBJX) check.$(OBJX) $(LIBS) -o ip14.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip14
	ip14.$(EXESUFFIX)

verify: ;

ip14.run: run

