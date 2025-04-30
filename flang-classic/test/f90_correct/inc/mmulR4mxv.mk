#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR4mxv  ########


mmulR4mxv: run
	

build:  $(SRC)/mmulR4mxv.f90
	-$(RM) mmulR4mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR4mxv.f90 -o mmulR4mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR4mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR4mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR4mxv
	mmulR4mxv.$(EXESUFFIX)

verify: ;

mmulR4mxv.run: run

