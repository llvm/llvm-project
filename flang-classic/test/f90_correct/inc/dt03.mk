#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt03  ########


dt03: run
	

build:  $(SRC)/dt03.f90
	-$(RM) dt03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt03.f90 -o dt03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt03.$(OBJX) check.$(OBJX) $(LIBS) -o dt03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt03
	dt03.$(EXESUFFIX)

verify: ;

dt03.run: run

