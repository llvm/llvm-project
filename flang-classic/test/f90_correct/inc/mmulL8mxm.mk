#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL8mxm  ########


mmulL8mxm: run
	

build:  $(SRC)/mmulL8mxm.f90
	-$(RM) mmulL8mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL8mxm.f90 -o mmulL8mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL8mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL8mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL8mxm
	mmulL8mxm.$(EXESUFFIX)

verify: ;

mmulL8mxm.run: run

