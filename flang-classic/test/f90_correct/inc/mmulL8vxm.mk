#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL8vxm  ########


mmulL8vxm: run
	

build:  $(SRC)/mmulL8vxm.f90
	-$(RM) mmulL8vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL8vxm.f90 -o mmulL8vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL8vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL8vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL8vxm
	mmulL8vxm.$(EXESUFFIX)

verify: ;

mmulL8vxm.run: run

