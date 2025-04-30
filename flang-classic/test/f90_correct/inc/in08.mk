#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in08  ########


in08: run
	

build:  $(SRC)/in08.f90
	-$(RM) in08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in08.f90 -o in08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in08.$(OBJX) check.$(OBJX) $(LIBS) -o in08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in08
	in08.$(EXESUFFIX)

verify: ;

in08.run: run

