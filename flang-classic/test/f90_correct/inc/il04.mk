#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test il04  ########


il04: run
	

build:  $(SRC)/il04.f90
	-$(RM) il04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/il04.f90 -o il04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) il04.$(OBJX) check.$(OBJX) $(LIBS) -o il04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test il04
	il04.$(EXESUFFIX)

verify: ;

il04.run: run

