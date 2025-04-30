#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI1mxv  ########


mmulI1mxv: run
	

build:  $(SRC)/mmulI1mxv.f90
	-$(RM) mmulI1mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI1mxv.f90 -o mmulI1mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI1mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI1mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI1mxv
	mmulI1mxv.$(EXESUFFIX)

verify: ;

mmulI1mxv.run: run

