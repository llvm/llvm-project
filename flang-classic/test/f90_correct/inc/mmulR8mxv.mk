#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR8mxv  ########


mmulR8mxv: run
	

build:  $(SRC)/mmulR8mxv.f90
	-$(RM) mmulR8mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR8mxv.f90 -o mmulR8mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR8mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR8mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR8mxv
	mmulR8mxv.$(EXESUFFIX)

verify: ;

mmulR8mxv.run: run

