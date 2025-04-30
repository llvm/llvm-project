#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulC8mxm  ########


mmulC8mxm: run
	

build:  $(SRC)/mmulC8mxm.f90
	-$(RM) mmulC8mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulC8mxm.f90 -o mmulC8mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulC8mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulC8mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulC8mxm
	mmulC8mxm.$(EXESUFFIX)

verify: ;

mmulC8mxm.run: run

