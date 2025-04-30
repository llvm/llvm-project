#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test il02  ########


il02: run
	

build:  $(SRC)/il02.f90
	-$(RM) il02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/il02.f90 -o il02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) il02.$(OBJX) check.$(OBJX) $(LIBS) -o il02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test il02
	il02.$(EXESUFFIX)

verify: ;

il02.run: run

