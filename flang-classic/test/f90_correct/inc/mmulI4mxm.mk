#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI4mxm  ########


mmulI4mxm: run
	

build:  $(SRC)/mmulI4mxm.f90
	-$(RM) mmulI4mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI4mxm.f90 -o mmulI4mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI4mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI4mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI4mxm
	mmulI4mxm.$(EXESUFFIX)

verify: ;

mmulI4mxm.run: run

