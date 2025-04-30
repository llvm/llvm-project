#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulC16mxm  ########


mmulC16mxm: run
	

build:  $(SRC)/mmulC16mxm.f90
	-$(RM) mmulC16mxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulC16mxm.f90 -o mmulC16mxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulC16mxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulC16mxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulC16mxm
	mmulC16mxm.$(EXESUFFIX)

verify: ;

mmulC16mxm.run: run

