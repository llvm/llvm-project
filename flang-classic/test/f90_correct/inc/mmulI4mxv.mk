#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI4mxv  ########


mmulI4mxv: run
	

build:  $(SRC)/mmulI4mxv.f90
	-$(RM) mmulI4mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI4mxv.f90 -o mmulI4mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI4mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI4mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI4mxv
	mmulI4mxv.$(EXESUFFIX)

verify: ;

mmulI4mxv.run: run

