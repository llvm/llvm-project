#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI2mxm  ########


mmulI2mxm: run
	

build:  $(SRC)/mmulI2mxm.f90
	-$(RM) mmulI2mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI2mxm.f90 -o mmulI2mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI2mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI2mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI2mxm
	mmulI2mxm.$(EXESUFFIX)

verify: ;

mmulI2mxm.run: run

