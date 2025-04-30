#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL4mxm  ########


mmulL4mxm: run
	

build:  $(SRC)/mmulL4mxm.f90
	-$(RM) mmulL4mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL4mxm.f90 -o mmulL4mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL4mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL4mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL4mxm
	mmulL4mxm.$(EXESUFFIX)

verify: ;

mmulL4mxm.run: run

