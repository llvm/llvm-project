#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL1mxv  ########


mmulL1mxv: run
	

build:  $(SRC)/mmulL1mxv.f90
	-$(RM) mmulL1mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL1mxv.f90 -o mmulL1mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL1mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL1mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL1mxv
	mmulL1mxv.$(EXESUFFIX)

verify: ;

mmulL1mxv.run: run

