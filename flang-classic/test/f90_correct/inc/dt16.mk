#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt16  ########


dt16: run
	

build:  $(SRC)/dt16.f90
	-$(RM) dt16.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt16.f90 -o dt16.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt16.$(OBJX) check.$(OBJX) $(LIBS) -o dt16.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt16
	dt16.$(EXESUFFIX)

verify: ;

dt16.run: run

