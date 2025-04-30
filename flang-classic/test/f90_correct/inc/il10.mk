#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test il10  ########


il10: run
	

build:  $(SRC)/il10.f90
	-$(RM) il10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/il10.f90 -o il10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) il10.$(OBJX) check.$(OBJX) $(LIBS) -o il10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test il10
	il10.$(EXESUFFIX)

verify: ;

il10.run: run

