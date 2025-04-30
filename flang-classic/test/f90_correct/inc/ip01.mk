#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip01  ########


ip01: run
	

build:  $(SRC)/ip01.f90
	-$(RM) ip01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip01.f90 -o ip01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip01.$(OBJX) check.$(OBJX) $(LIBS) -o ip01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip01
	ip01.$(EXESUFFIX)

verify: ;

ip01.run: run

