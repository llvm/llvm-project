#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ae00  ########


ae00: run
	

build:  $(SRC)/ae00.f
	-$(RM) ae00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ae00.f -o ae00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ae00.$(OBJX) check.$(OBJX) $(LIBS) -o ae00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ae00
	ae00.$(EXESUFFIX)

verify: ;

ae00.run: run

