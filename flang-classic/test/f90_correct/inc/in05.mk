#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in05  ########


in05: run
	

build:  $(SRC)/in05.f90
	-$(RM) in05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in05.f90 -o in05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in05.$(OBJX) check.$(OBJX) $(LIBS) -o in05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in05
	in05.$(EXESUFFIX)

verify: ;

in05.run: run

