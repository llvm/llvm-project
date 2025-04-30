#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip12  ########


ip12: run
	

build:  $(SRC)/ip12.f90
	-$(RM) ip12.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip12.f90 -o ip12.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip12.$(OBJX) check.$(OBJX) $(LIBS) -o ip12.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip12
	ip12.$(EXESUFFIX)

verify: ;

ip12.run: run

