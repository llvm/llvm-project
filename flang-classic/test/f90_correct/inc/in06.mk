#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in06  ########


in06: run
	

build:  $(SRC)/in06.f90
	-$(RM) in06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in06.f90 -o in06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in06.$(OBJX) check.$(OBJX) $(LIBS) -o in06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in06
	in06.$(EXESUFFIX)

verify: ;

in06.run: run

