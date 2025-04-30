#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI8mxm  ########


mmulI8mxm: run
	

build:  $(SRC)/mmulI8mxm.f90
	-$(RM) mmulI8mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI8mxm.f90 -o mmulI8mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI8mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI8mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI8mxm
	mmulI8mxm.$(EXESUFFIX)

verify: ;

mmulI8mxm.run: run

