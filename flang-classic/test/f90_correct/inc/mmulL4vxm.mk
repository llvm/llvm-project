#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulL4vxm  ########


mmulL4vxm: run
	

build:  $(SRC)/mmulL4vxm.f90
	-$(RM) mmulL4vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulL4vxm.f90 -o mmulL4vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulL4vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulL4vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulL4vxm
	mmulL4vxm.$(EXESUFFIX)

verify: ;

mmulL4vxm.run: run

