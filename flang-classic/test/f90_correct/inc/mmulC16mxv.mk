#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulC16mxv  ########


mmulC16mxv: run
	

build:  $(SRC)/mmulC16mxv.f90
	-$(RM) mmulC16mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulC16mxv.f90 -o mmulC16mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulC16mxv.$(OBJX) check.$(OBJX) $(LIBS) -o mmulC16mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulC16mxv
	mmulC16mxv.$(EXESUFFIX)

verify: ;

mmulC16mxv.run: run

