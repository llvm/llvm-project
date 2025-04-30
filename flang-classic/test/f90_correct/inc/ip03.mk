#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip03  ########


ip03: run
	

build:  $(SRC)/ip03.f90
	-$(RM) ip03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip03.f90 -o ip03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip03.$(OBJX) check.$(OBJX) $(LIBS) -o ip03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip03
	ip03.$(EXESUFFIX)

verify: ;

ip03.run: run

