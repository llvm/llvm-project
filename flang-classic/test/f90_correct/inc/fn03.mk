#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fn03  ########


fn03: run
	

build:  $(SRC)/fn03.f90
	-$(RM) fn03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fn03.f90 -o fn03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fn03.$(OBJX) check.$(OBJX) $(LIBS) -o fn03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fn03
	fn03.$(EXESUFFIX)

verify: ;

fn03.run: run

