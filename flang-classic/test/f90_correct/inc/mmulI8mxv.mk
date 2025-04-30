#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI8mxv  ########


mmulI8mxv: run
	

build:  $(SRC)/mmulI8mxv.f90
	-$(RM) mmulI8mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI8mxv.f90 -o mmulI8mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI8mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI8mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI8mxv
	mmulI8mxv.$(EXESUFFIX)

verify: ;

mmulI8mxv.run: run

