#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL2vxm  ########


mmulL2vxm: run
	

build:  $(SRC)/mmulL2vxm.f90
	-$(RM) mmulL2vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL2vxm.f90 -o mmulL2vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL2vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL2vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL2vxm
	mmulL2vxm.$(EXESUFFIX)

verify: ;

mmulL2vxm.run: run

