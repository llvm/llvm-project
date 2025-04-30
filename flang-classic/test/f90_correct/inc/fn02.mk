#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fn02  ########


fn02: run
	

build:  $(SRC)/fn02.f90
	-$(RM) fn02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fn02.f90 -o fn02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fn02.$(OBJX) check.$(OBJX) $(LIBS) -o fn02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fn02
	fn02.$(EXESUFFIX)

verify: ;

fn02.run: run

