#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test il08  ########


il08: run
	

build:  $(SRC)/il08.f90
	-$(RM) il08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/il08.f90 -o il08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) il08.$(OBJX) check.$(OBJX) $(LIBS) -o il08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test il08
	il08.$(EXESUFFIX)

verify: ;

il08.run: run

