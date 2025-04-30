#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gb00  ########


gb00: run
	

build:  $(SRC)/gb00.f
	-$(RM) gb00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gb00.f -o gb00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gb00.$(OBJX) check.$(OBJX) $(LIBS) -o gb00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gb00
	gb00.$(EXESUFFIX)

verify: ;

gb00.run: run

