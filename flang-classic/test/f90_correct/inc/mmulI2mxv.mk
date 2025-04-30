#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI2mxv  ########


mmulI2mxv: run
	

build:  $(SRC)/mmulI2mxv.f90
	-$(RM) mmulI2mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI2mxv.f90 -o mmulI2mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI2mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI2mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI2mxv
	mmulI2mxv.$(EXESUFFIX)

verify: ;

mmulI2mxv.run: run

