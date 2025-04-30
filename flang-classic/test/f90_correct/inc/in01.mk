#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in01  ########


in01: run
	

build:  $(SRC)/in01.f90
	-$(RM) in01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in01.f90 -o in01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in01.$(OBJX) check.$(OBJX) $(LIBS) -o in01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in01
	in01.$(EXESUFFIX)

verify: ;

in01.run: run

