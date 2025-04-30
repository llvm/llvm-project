#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test il01  ########


il01: run
	

build:  $(SRC)/il01.f90
	-$(RM) il01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/il01.f90 -o il01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) il01.$(OBJX) check.$(OBJX) $(LIBS) -o il01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test il01
	il01.$(EXESUFFIX)

verify: ;

il01.run: run

