#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL1mxm  ########


mmulL1mxm: run
	

build:  $(SRC)/mmulL1mxm.f90
	-$(RM) mmulL1mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL1mxm.f90 -o mmulL1mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL1mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL1mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL1mxm
	mmulL1mxm.$(EXESUFFIX)

verify: ;

mmulL1mxm.run: run

