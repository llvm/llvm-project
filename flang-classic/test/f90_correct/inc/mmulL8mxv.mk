#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL8mxv  ########


mmulL8mxv: run
	

build:  $(SRC)/mmulL8mxv.f90
	-$(RM) mmulL8mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL8mxv.f90 -o mmulL8mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL8mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL8mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL8mxv
	mmulL8mxv.$(EXESUFFIX)

verify: ;

mmulL8mxv.run: run

