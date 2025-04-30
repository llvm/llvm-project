#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fn01  ########


fn01: run
	

build:  $(SRC)/fn01.f90
	-$(RM) fn01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fn01.f90 -o fn01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fn01.$(OBJX) check.$(OBJX) $(LIBS) -o fn01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fn01
	fn01.$(EXESUFFIX)

verify: ;

fn01.run: run

