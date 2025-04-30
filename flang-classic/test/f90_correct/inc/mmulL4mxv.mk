#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL4mxv  ########


mmulL4mxv: run
	

build:  $(SRC)/mmulL4mxv.f90
	-$(RM) mmulL4mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL4mxv.f90 -o mmulL4mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL4mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL4mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL4mxv
	mmulL4mxv.$(EXESUFFIX)

verify: ;

mmulL4mxv.run: run

