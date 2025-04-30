#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip40  ########


ip40: run
	

build:  $(SRC)/ip40.f90
	-$(RM) ip40.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip40.f90 -o ip40.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip40.$(OBJX) check.$(OBJX) $(LIBS) -o ip40.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip40
	ip40.$(EXESUFFIX)

verify: ;

ip40.run: run

