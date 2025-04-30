#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR8mxm  ########


mmulR8mxm: run
	

build:  $(SRC)/mmulR8mxm.f90
	-$(RM) mmulR8mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR8mxm.f90 -o mmulR8mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR8mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR8mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR8mxm
	mmulR8mxm.$(EXESUFFIX)

verify: ;

mmulR8mxm.run: run

