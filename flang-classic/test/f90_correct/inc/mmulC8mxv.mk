#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulC8mxv  ########


mmulC8mxv: run
	

build:  $(SRC)/mmulC8mxv.f90
	-$(RM) mmulC8mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulC8mxv.f90 -o mmulC8mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulC8mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulC8mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulC8mxv
	mmulC8mxv.$(EXESUFFIX)

verify: ;

mmulC8mxv.run: run

