#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip41  ########


ip41: run
	

build:  $(SRC)/ip41.f90
	-$(RM) ip41.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip41.f90 -o ip41.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip41.$(OBJX) check.$(OBJX) $(LIBS) -o ip41.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip41
	ip41.$(EXESUFFIX)

verify: ;

ip41.run: run

