#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test il06  ########


il06: run
	

build:  $(SRC)/il06.f90
	-$(RM) il06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/il06.f90 -o il06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) il06.$(OBJX) check.$(OBJX) $(LIBS) -o il06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test il06
	il06.$(EXESUFFIX)

verify: ;

il06.run: run

