#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip19  ########


ip19: run
	

build:  $(SRC)/ip19.f90
	-$(RM) ip19.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip19.f90 -o ip19.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip19.$(OBJX) check.$(OBJX) $(LIBS) -o ip19.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip19
	ip19.$(EXESUFFIX)

verify: ;

ip19.run: run

