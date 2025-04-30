#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL2mxm  ########


mmulL2mxm: run
	

build:  $(SRC)/mmulL2mxm.f90
	-$(RM) mmulL2mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL2mxm.f90 -o mmulL2mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL2mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL2mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL2mxm
	mmulL2mxm.$(EXESUFFIX)

verify: ;

mmulL2mxm.run: run

