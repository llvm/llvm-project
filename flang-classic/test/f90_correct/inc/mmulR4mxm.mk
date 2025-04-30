#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR4mxm  ########


mmulR4mxm: run
	

build:  $(SRC)/mmulR4mxm.f90
	-$(RM) mmulR4mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR4mxm.f90 -o mmulR4mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR4mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR4mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR4mxm
	mmulR4mxm.$(EXESUFFIX)

verify: ;

mmulR4mxm.run: run

