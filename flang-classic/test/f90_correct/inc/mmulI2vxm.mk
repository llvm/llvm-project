#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI2vxm  ########


mmulI2vxm: run
	

build:  $(SRC)/mmulI2vxm.f90
	-$(RM) mmulI2vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI2vxm.f90 -o mmulI2vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI2vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI2vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI2vxm
	mmulI2vxm.$(EXESUFFIX)

verify: ;

mmulI2vxm.run: run

