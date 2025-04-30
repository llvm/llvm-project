#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test na25  ########


na25: run
	

build:  $(SRC)/na25.f90
	-$(RM) na25.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/na25.f90 -o na25.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) na25.$(OBJX) check.$(OBJX) $(LIBS) -o na25.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test na25
	na25.$(EXESUFFIX)

verify: ;

na25.run: run

