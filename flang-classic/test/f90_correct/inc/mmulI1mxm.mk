#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI1mxm  ########


mmulI1mxm: run
	

build:  $(SRC)/mmulI1mxm.f90
	-$(RM) mmulI1mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI1mxm.f90 -o mmulI1mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI1mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI1mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI1mxm
	mmulI1mxm.$(EXESUFFIX)

verify: ;

mmulI1mxm.run: run

