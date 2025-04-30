#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ib40  ########


ib40: run
	

build:  $(SRC)/ib40.f
	-$(RM) ib40.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ib40.f -o ib40.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ib40.$(OBJX) check.$(OBJX) $(LIBS) -o ib40.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ib40
	ib40.$(EXESUFFIX)

verify: ;

ib40.run: run

