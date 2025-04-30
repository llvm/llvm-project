#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL1vxm  ########


mmulL1vxm: run
	

build:  $(SRC)/mmulL1vxm.f90
	-$(RM) mmulL1vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL1vxm.f90 -o mmulL1vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL1vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL1vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL1vxm
	mmulL1vxm.$(EXESUFFIX)

verify: ;

mmulL1vxm.run: run

